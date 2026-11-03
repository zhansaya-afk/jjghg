"""Персонализатор: маршрутизатор → поиск по базе знаний → (необязательно) локальная LLM → страж.

answer(question, facts, history) — то, что вызывает экзамен (scripts/eval_personalizer.py).
personalize(...) — вход для ноутбука: сам собирает карточку фактов из результата ядра.
"""

import re

from ..personalizer_eval import violations
from . import texts as T
from .facts import build_facts, hb_from_question, photo_ok
from .retrieve import SLOT, default_retriever, load_kb
from .router import lang_of, needs_retake, route

THRESHOLD = 0.40  # сходство вопроса с базой ниже порога → «не знаю»; на dev самое низкое верное 0,44
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
    return "answer", T.kk_explain(facts, question)


def rules_answer(question, facts, history=None, retriever=None):
    """Режим rules: без языковой модели. → {'text', 'mode', 'sources', 'lang'}."""
    q = (question or "").strip()
    lang = lang_of(q)
    mode, why = route(q, facts, history)
    sources = []
    if lang == "kk":
        mode, text = _kk(mode, why, q, facts)
        return {"text": tidy(text), "mode": mode, "sources": sources, "lang": lang}
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
        else:
            hits = (retriever or default_retriever("ru")).search(q, k=1)
            fr = hits[0][0] if hits and hits[0][1] >= THRESHOLD else None
        if hb_from_question(q, facts) and (fr is None or fr.id in NOT_ABOUT_NUMBER):
            # «у меня 9,2», «гемоглобин 160 это много?»: отвечать про число; есть фото → сравнить с ним
            fr = _fragment("photo_vs_lab" if photo_ok(facts) else LAB)
        if fr is None and q and ABOUT_PHOTO.search(q) and not needs_retake(q, facts):
            fr = _fragment(
                EXPLAIN
            )  # вопрос про «мой риск / результат», не нашёлся дословно → объяснить результат
        retake = needs_retake(q, facts)
        if retake and (fr.id in ABOUT_RESULT if fr else ABOUT_PHOTO.search(q)):
            mode, text = "retake", T.retake(facts)
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
        if retake and mode != "retake":
            # общий вопрос при непригодном снимке: ответ + напоминание переснять
            text = text + " " + T.RETAKE_NOTE
    return {"text": tidy(text), "mode": mode, "sources": sources, "lang": lang}


def answer(question, facts, history=None, *, engine="rules", llm=None, retriever=None):
    """Ответ на вопрос по карточке фактов. engine='llm' + llm=callable(messages)→str: модель только
    переформулирует ответ режима rules; при любом нарушении стража остаётся ответ режима rules."""
    base = rules_answer(question, facts, history, retriever)
    out = dict(base, engine="rules", guard={})
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
    facts = build_facts(result, lab, profile, survey)
    return dict(answer(question, facts, history, **kw), facts=facts)
