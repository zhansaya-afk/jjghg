"""Персонализатор: маршрутизатор → поиск по базе знаний → (необязательно) локальная LLM → страж.

answer(question, facts, history) — то, что вызывает экзамен (scripts/eval_personalizer.py).
personalize(...) — вход для ноутбука: сам собирает карточку фактов из результата ядра.
"""

import re

from ..personalizer_eval import violations
from . import texts as T
from .context import (
    asks_dose,
    base_question,
    context_query,
    dialog_facts,
    is_follow_up,
    refers_back,
)
from .facts import build_facts, hb_from_question, photo_ok
from .retrieve import SLOT, default_retriever, load_kb
from .router import lang_of, needs_retake, route

THRESHOLD = 0.35  # сходство ниже порога → «не знаю»; выбран на наборе A (evals/personalizer/search_questions.json)
# вопрос без слов про анемию («На парах выдыхаюсь»): маршрутизатор считает его «не моей темой»; если поиск
# всё же уверенно находит тему (сходство не ниже этого порога), отвечаем ей. Порог выбран на наборах A и P.
OFF_DOMAIN_THRESHOLD = 0.45
LLM_TOP_K = 3  # нейросеть выбирает тему только среди стольких лучших тем поиска (или «нет темы»)
OFF_DOMAIN_NOTE = "нет слов про анемию, но поиск уверенно нашёл тему"
REFER_MARGIN = 0.25  # «это» в разговоре: тема по прошлому вопросу берётся, если сходство выше на столько
EXPLAIN = "explain_result"
LAB = "lab_explain"
# фрагменты о самом результате фото: при непригодном снимке вместо них — «переснимите»
ABOUT_RESULT = {
    "explain_result",
    "clarify_risk",
    "low_risk_meaning",
    "probability",
    "why_tissue",
    "feel_fine_high",
    "hb_by_photo",
    "photo_vs_lab",
    "next_steps",
    "accuracy",
    "photo_howto",
    "photo_conditions",
    "need_test",
}
# вопрос не нашёлся в базе, но он про снимок / результат → при непригодном снимке «переснимите»
ABOUT_PHOTO = re.compile(
    r"результат|фото|снимок|снимк|оценк|риск|гемоглоб|почему\s+нет", re.IGNORECASE
)
# общие фрагменты: если в вопросе своё число Hb, они на него не отвечают → объяснение этого числа
NOT_ABOUT_NUMBER = {
    "explain_result",
    "clarify_risk",
    "low_risk_meaning",
    "probability",
    "hb_by_photo",
    "norm",
    "what_is_anemia",
    "causes",
    "symptoms",
    "serious",
    "next_steps",
}
# «Я беременна, опасна ли анемия для малыша?»: общий ответ про опасность без беременности — мимо
PREGNANT_Q = re.compile(r"беремен|жду\s+(?:реб[её]нка|малыша)", re.IGNORECASE)
GENERAL_FOR_PREGNANCY = {"serious", "what_is_anemia"}
AGE_Q = re.compile(r"\bмне\s+\d{1,3}\s*(?:лет|год)", re.IGNORECASE)
SAFE_FALLBACK = (
    "Не могу ответить на это надёжно. Это скрининг, не диагноз: покажите результат врачу-терапевту, он подскажет, "
    "какие анализы нужны."
)


def fill(body, facts, question=""):
    text = SLOT.sub(lambda m: T.SLOTS[m.group(1)](facts, question), body)
    return tidy(text)


def tidy(text):
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+([.,:;!?])", r"\1", text)


def _fragment(fid):
    return next(fr for fr in load_kb("ru") if fr.id == fid)


def _kk(mode, why, question, facts):
    if mode == "urgent":
        return "urgent", T.kk_urgent(facts, why)
    if mode == "child":
        return "child", T.KK_CHILD
    if why == "dose":
        return "answer", T.KK_DOSE
    if mode == "retake":
        return "retake", T.KK_RETAKE
    if mode in ("out_of_scope", "dont_know"):
        return "out_of_scope", T.KK_OUT
    if why == "drug":
        return "answer", T.KK_DOSE
    if why is None and needs_retake(question, facts):  # снимок непригоден, анализа нет
        if T.kk_about_result(question, facts):
            return "retake", T.KK_RETAKE
        return "answer", T.join(T.KK_ONLY_RESULT, T.KK_RETAKE)
    if why is None and not T.kk_about_result(question, facts):
        # базы знаний на казахском нет: честно сказать и объяснить результат (вопрос «что есть?» не о нём)
        return "answer", T.join(T.KK_ONLY_RESULT, T.kk_explain(facts, question))
    return "answer", T.kk_explain(facts, question)


def rules_answer(
    question, facts, history=None, retriever=None, force=None, search=None, pick=None
):
    """Режим rules: без языковой модели. → {'text', 'mode', 'sources', 'lang', 'why', 'trace'}.
    force=(режим, причина) — маршрут уже решён снаружи (например, «А сколько?» после вопроса о железе).
    search — текст для поиска темы, если он не question: у «А при беременности?» тема — из прошлого вопроса.
    trace — как найден ответ: правило маршрутизатора или лучшие темы поиска со сходством (для Я5 и Я6).
    pick — id темы, уже выбранной снаружи (нейросеть поняла перефразированный вопрос): поиск не нужен."""
    q = (question or "").strip()
    lang = lang_of(q)
    mode, why = route(q, facts, history)
    if force and mode != "urgent":
        mode, why = force
    sources = []
    off_domain = None
    if mode == "out_of_scope" and why is None and lang == "ru" and not pick and q:
        # «не моя тема» только по словам: если поиск уверенно находит тему базы — это вопрос о самочувствии
        # или еде без слова «анемия». Срочное, дозы, препараты, диагноз уже решены правилами выше.
        off_domain = (retriever or default_retriever("ru")).search(search or q, k=3)
        if off_domain and off_domain[0][1] >= OFF_DOMAIN_THRESHOLD:
            mode = None
    trace = {
        "rule": (why or mode) if (mode or why) else None,
        "search": [],
        "notes": [],
    }
    note = trace["notes"].append
    if off_domain is not None:
        trace["search"] = [(h.id, round(sc, 2)) for h, sc in off_domain]
        if mode is None:
            note(OFF_DOMAIN_NOTE)
    if lang == "kk":
        mode, text = _kk(mode, why, q, facts)
        note("казахский: готовый шаблон")
        return {
            "text": tidy(text),
            "mode": mode,
            "sources": sources,
            "lang": lang,
            "why": why,
            "trace": trace,
        }
    if mode == "urgent":
        text = T.urgent(facts, why)
    elif mode == "child":
        text = T.CHILD
    elif mode == "dont_know":
        text = T.PRICE if why == "price" else T.past_data(facts)
    elif mode == "out_of_scope":
        text = T.OUT_OF_SCOPE
    elif why == "greeting":
        text = T.GREETING
    elif why == "thanks":
        text = T.THANKS
    elif why == "about":
        text = T.ABOUT
    elif why == "language":
        text = T.ONLY_RU_KK
    elif why == "certificate":
        text = T.CERTIFICATE
    elif why == "dose":
        text = T.dose(facts)
    elif why == "drug":
        text = T.DRUG
    elif why == "diagnosis":
        text = T.diagnosis(facts, q)
    else:
        mode = "answer"
        if not q:
            fr = _fragment(EXPLAIN)
            note("пустой вопрос: объяснение результата")
        elif pick:
            fr = _fragment(pick)
            note(f"нейросеть поняла вопрос: тема {pick}")
        else:
            hits = (retriever or default_retriever("ru")).search(search or q, k=3)
            trace["search"] = [(h.id, round(sc, 2)) for h, sc in hits]
            fr = hits[0][0] if hits and hits[0][1] >= THRESHOLD else None
            if (
                fr is not None
                and fr.id in GENERAL_FOR_PREGNANCY
                and PREGNANT_Q.search(q)
            ):
                fr = _fragment("pregnancy")
                note("в вопросе беременность: тема pregnancy вместо общей")
        if hb_from_question(q, facts) and (fr is None or fr.id in NOT_ABOUT_NUMBER):
            # «у меня 9,2», «гемоглобин 160 это много?»: отвечать про число; есть фото → сравнить с ним
            fr = _fragment("photo_vs_lab" if photo_ok(facts) else LAB)
            note(f"в вопросе число Hb: тема {fr.id}")
        if fr is not None and fr.id == LAB and facts.get("photo") and facts.get("lab"):
            # есть и фото, и анализ: объяснить анализ рядом с фото
            fr = _fragment("photo_vs_lab")
            note("есть и фото, и анализ: тема photo_vs_lab")
        if fr is None and AGE_Q.search(q):
            # «Мне 16 лет, это считается?» — подходит ли проверка по возрасту
            fr = _fragment("age_scope")
            note("в вопросе возраст: тема age_scope")
        if fr is None and q and ABOUT_PHOTO.search(q) and not needs_retake(q, facts):
            # вопрос про «мой риск / результат», не нашёлся дословно → объяснить результат
            fr = _fragment(EXPLAIN)
            note("вопрос про результат: тема explain_result")
        retake = needs_retake(q, facts)
        if retake and (fr.id in ABOUT_RESULT if fr else ABOUT_PHOTO.search(q)):
            mode, text = "retake", T.retake(facts)
            note("снимок непригоден: «переснимите»")
        elif fr is None:
            mode, text = "dont_know", T.DONT_KNOW
        else:
            text, sources = fill(fr.body, facts, q), [fr.id]
            data = [k for k in fr.slots if k != "doctor"]
            static = SLOT.sub("", fr.body).strip()
            if (
                data
                and len(static) < 60
                and not any(T.SLOTS[k](facts, q) for k in data)
            ):
                # фрагмент — почти одни факты, а их нет (например, анализа нет) → общее объяснение
                text, sources = fill(_fragment(EXPLAIN).body, facts, q), [EXPLAIN]
                note(f"для темы {fr.id} нет данных: общее объяснение результата")
        if retake and mode != "retake":
            # общий вопрос при непригодном снимке: ответ + напоминание переснять
            text = text + " " + T.RETAKE_NOTE
    return {
        "text": tidy(text),
        "mode": mode,
        "sources": sources,
        "lang": lang,
        "why": why,
        "trace": trace,
    }


def answer(question, facts, history=None, *, engine="rules", llm=None, retriever=None):
    """Ответ на вопрос по карточке фактов. llm=callable(messages)→str — локальная модель:
    engine='understand' — модель только выбирает тему базы для вопроса, который поиск не нашёл
    (перефраз); ответ — проверенный текст базы. engine='llm' — то же + модель переформулирует ответ;
    при любом нарушении стража остаётся ответ режима rules.

    Память разговора (только history этого запуска): уточнение «А для мужчин?» ищется вместе с прошлым
    вопросом (out['context']); сказанное о себе («я вегетарианка») меняет советы (out['facts'])."""
    q = (question or "").strip()
    facts = dialog_facts(facts, history, q)
    cq = context_query(q, history)
    force = ("answer", "dose") if asks_dose(q, cq) else None
    # уточнение без своей темы: тема — из прошлого вопроса, условие («для мужчин») — в подстановке фактов
    topic = base_question(history) if cq != q else None
    base = rules_answer(cq, facts, history, retriever, force, topic)
    lost = base["why"] is None and base["mode"] in ("out_of_scope", "dont_know")
    clarify = False
    if cq == q and lost:
        # «А ферритин?» сам не нашёлся → как уточнение прошлого вопроса
        wq = context_query(q, history, weak=True)
        alt = rules_answer(wq, facts, history, retriever) if wq != q else None
        if alt and alt["mode"] not in ("out_of_scope", "dont_know"):
            base, cq = alt, wq
        elif is_follow_up(q) and base["lang"] == "ru":
            clarify = True  # «А для мужчин?» без прошлого вопроса
    if cq == q and not clarify and base["why"] is None and refers_back(q):
        base = _refer_back(q, facts, history, retriever, base)
    if clarify:
        base = dict(base, mode="dont_know", text=T.CLARIFY, sources=[])
        base["trace"] = dict(
            base["trace"],
            notes=[*base["trace"]["notes"], "уточнение без прошлого вопроса"],
        )
    if force:
        base["trace"] = dict(
            base["trace"],
            notes=[*base["trace"]["notes"], "«сколько» после вопроса о лекарстве"],
        )
    picked = None
    if (
        engine in ("understand", "llm")
        and llm is not None
        and lost
        and not clarify
        and base["mode"] in ("out_of_scope", "dont_know")
        and base["lang"] == "ru"
    ):
        # поиск по буквам не нашёл (перефраз, синонимы): модель выбирает тему по смыслу или «нет темы».
        # Срочные признаки и отказы (дозы, препараты, диагноз, дети) решены правилами раньше: lost = why is None
        from .generate import pick_topic

        # выбор только из LLM_TOP_K лучших тем поиска: на перефразах 30 вопросов с полным списком тем
        # модель чаще уходила в чужую тему (не та тема 2 → 6 из 30, прогон 10.11)
        hits = (retriever or default_retriever("ru")).search(cq, k=LLM_TOP_K)
        candidates = [fr for fr, sc in hits if sc > 0]
        picked = pick_topic(llm, cq, candidates) if candidates else None
        if picked:
            first = base["trace"]
            why_lost = (
                "слова вопроса не узнаны как тема анемии"
                if base["mode"] == "out_of_scope"
                else "поиск по словам не набрал порог"
            )
            force_mode = ("answer", None) if base["mode"] == "out_of_scope" else None
            base = rules_answer(cq, facts, history, retriever, force_mode, pick=picked)
            base["trace"] = dict(
                base["trace"],
                rule=None,
                search=first["search"],
                notes=[
                    f"{why_lost}; нейросеть поняла вопрос: тема {picked}",
                    *[
                        n
                        for n in base["trace"]["notes"]
                        if not n.startswith("нейросеть")
                    ],
                ],
            )
    question = cq
    out = dict(
        base,
        engine="understand" if picked else "rules",
        guard={},
        context=cq if cq != q else None,
        facts=facts,
    )
    if (
        engine == "llm"
        and llm is not None
        and base["mode"] == "answer"
        and base["sources"]
    ):
        from .generate import rewrite

        text = rewrite(llm, question, facts, base, history)
        bad = (
            violations(text, facts, question, base["lang"]) if text else {"пусто": True}
        )
        bad.update(_lost_facts(base["text"], text or ""))
        if bad:
            out["guard"] = bad
        else:
            out.update(text=tidy(text), engine="llm")
    bad = violations(out["text"], facts, question, out["lang"])
    if bad:  # не должно случаться в режиме rules: это ошибка шаблона, ответ заменяется безопасным
        out.update(text=SAFE_FALLBACK, guard={**out["guard"], **bad}, sources=[])
    return out


def _top(out):
    hits = (out.get("trace") or {}).get("search") or []
    return (
        hits[0][1] if hits and out["mode"] not in ("out_of_scope", "dont_know") else 0.0
    )


def _refer_back(q, facts, history, retriever, base):
    """«Можно ли это предотвратить?», «Зачем его сдавать?» в разговоре об анемии. Срабатывает, только если
    сам вопрос не нашёлся или нашёлся без слов про анемию: тема, найденная по словам вопроса, не меняется
    («Пью чай после обеда, это плохо?»). Разговор уже по теме, поэтому фильтр «не моя тема» не нужен.
    Вопрос вместе с прошлым берётся, если сам вопрос не нашёлся или прошлый даёт тему точнее на REFER_MARGIN
    («Что такое ферритин? → Зачем его сдавать?»). Правила (срочное, дозы, препараты, диагноз) уже проверены:
    сюда попадают только вопросы с why=None."""
    prev = base_question(history)
    if base["lang"] != "ru" or not prev or route(prev, facts)[0] == "out_of_scope":
        return base  # прошлый вопрос не об анемии («Как выучить химию? → Это сложно?»)
    weak = base["mode"] != "answer" or any(
        n.startswith(OFF_DOMAIN_NOTE) for n in base["trace"]["notes"]
    )
    if not weak:
        return base  # «Пью чай после обеда, это плохо?» — своя тема найдена по словам вопроса
    back = ("answer", None)
    alone = (
        base
        if base["mode"] == "answer"
        else rules_answer(q, facts, history, retriever, back)
    )
    joined = f"{prev.rstrip('?.! ')}? {q}"
    alt = rules_answer(q, facts, history, retriever, back, search=joined)
    if _top(alt) and _top(alt) - _top(alone) >= (REFER_MARGIN if _top(alone) else 0):
        note = f"«это» — о прошлом вопросе «{prev}»: искался вместе с ним"
    elif _top(alone) and alone is not base:
        alt, note = alone, "разговор об анемии: вопрос без слова «анемия» искался сам"
    else:
        return base
    alt["trace"] = dict(alt["trace"], notes=[note, *alt["trace"]["notes"]])
    return alt


def _lost_facts(base, text):
    """LLM потеряла цифры г/л, уровень риска или отсылку к врачу из ответа rules → нарушение."""
    out = {}
    nums = set(re.findall(r"\d{2,3}(?=\s*(?:–\s*\d{2,3}\s*)?г/л)", base))
    if nums - set(re.findall(r"\d{2,3}", text)):
        out["потеряны цифры"] = sorted(nums - set(re.findall(r"\d{2,3}", text)))
    for w in ("низк", "средн", "высок"):
        if re.search(rf"риск\w*\s+анемии\s+{w}", base) and w not in text:
            out["потерян риск"] = w
    if "врач" in base and "врач" not in text:
        out["потерян врач"] = True
    return out


def personalize(
    result=None, lab=None, profile=None, survey=None, question=None, history=None, **kw
):
    """Результат ядра (api.Result / analyze_photo) и/или анализ крови → ответ персонализатора."""
    return answer(question, build_facts(result, lab, profile, survey), history, **kw)
