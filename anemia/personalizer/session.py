"""Диалог персонализатора для ячейки Я5: объяснение результата → короткий опрос → вопросы пользователя.

Ввод и вывод передаются функциями (ask, say), поэтому тот же код работает в Colab (input / print)
и в тестах (поддельный ввод). Риск по фото и анализ крови не пересчитываются: персонализатор только
объясняет то, что вернули predict.analyze_photo (Я2) и labs.lab_hb_report (Я3).
"""

from .engine import answer
from .facts import build_facts
from .survey import questions_for, to_survey

YES = {"да", "д", "ага", "есть", "бывает", "yes", "y", "иә", "ия", "иа", "+", "1"}
NO = {"нет", "н", "неа", "не", "no", "n", "жоқ", "жок", "-", "0"}
TEXT = {
    "ru": {
        "survey": "Короткий опрос: ответьте «да» или «нет», Enter — пропустить вопрос.",
        "after": "Что делать дальше?",
        "ask": "Ваш вопрос (Enter — выход): ",
        "bye": "Я не ставлю диагноз: окончательный ответ дают анализ крови и врач.",
        "stop": "Из-за опасного признака дальше вопросы не разбираем: сначала обратитесь за помощью.",
        "explain": "Что значит мой результат?",
        "nothing": "Нет результата: сначала запустите ячейку Я2 (фото) или Я3 (анализ крови).",
    },
    "kk": {
        "survey": "Қысқа сауалнама: «иә» немесе «жоқ» деп жауап беріңіз, Enter — сұрақты өткізу.",
        "after": "Енді не істеу керек?",
        "ask": "Сұрағыңыз (Enter — шығу): ",
        "bye": "Мен диагноз қоймаймын: нақты жауапты қан талдауы мен дәрігер береді.",
        "stop": "Қауіпті белгіге байланысты әрі қарай сұрақтарды талдамаймыз: алдымен көмекке жүгініңіз.",
        "explain": "Нәтижем нені білдіреді?",
        "nothing": "Нәтиже жоқ: алдымен Я2 (фото) немесе Я3 (қан талдауы) ұяшығын іске қосыңыз.",
    },
}


def yes_no(text):
    """«да» / «нет» / пусто → True / False / None (не знаю, пропуск, непонятный ответ)."""
    t = (text or "").strip().lower().strip(".!")
    return True if t in YES else False if t in NO else None


def profile_from_form(sex="Не указан", age=None, pregnant="Не указано"):
    """Поля форм Я2 / Я3 → профиль карточки фактов."""
    return {
        "sex": {"Женский": "F", "Мужской": "M"}.get(sex),
        "age": age,
        "pregnant": {"Да": True, "Нет": False}.get(pregnant),
    }


def chat(
    result=None,
    lab=None,
    profile=None,
    *,
    ask=None,
    say=None,
    survey=True,
    lang="ru",
    max_turns=30,
    **kw,
):
    """Полный разговор. → {'facts', 'survey', 'history'}. kw уходят в engine.answer (engine='llm', llm=...)."""
    # input / print берутся в момент вызова, а не при определении функции (их подменяют тесты)
    ask = ask or (lambda prompt: input(prompt))
    say = say or (lambda text: print(text))
    tx = TEXT.get(lang, TEXT["ru"])
    if result is None and lab is None:
        say(tx["nothing"])
        return {"facts": None, "survey": None, "history": []}
    facts = build_facts(result, lab, profile)
    history = []

    def reply(question, shown=None):
        out = answer(question, facts, history, **kw)
        say(out["text"])
        history.append(
            {"role": "user", "text": shown if shown is not None else question}
        )
        history.append({"role": "assistant", "text": out["text"]})
        return out

    first = reply(tx["explain"] if lang == "kk" else "", shown=tx["explain"])
    if first["mode"] in ("urgent", "child"):
        return {"facts": facts, "survey": None, "history": history}

    answers = {}
    qs = questions_for(facts) if survey else []
    if qs:
        say("")
        say(tx["survey"])
        for q in qs:
            answers[q.key] = yes_no(ask(f"{getattr(q, lang, q.ru)} "))
            if q.key == "red_flag" and answers[q.key]:
                break  # опасный признак — остальное не спрашиваем
        facts = build_facts(result, lab, profile, to_survey(answers, facts))
        flagged = bool(answers.get("red_flag"))
        if (
            lang == "ru" or flagged
        ):  # для kk нет отдельного шаблона «что дальше» — только срочность
            say("")
            reply(tx["after"])
        if flagged:
            say(tx["stop"])
            return {"facts": facts, "survey": answers, "history": history}

    for _ in range(max_turns):
        say("")
        try:
            q = ask(tx["ask"]).strip()
        except EOFError:  # нет интерактивного ввода (запуск без браузера)
            break
        if not q:
            break
        reply(
            q
        )  # красный флаг в вопросе → ответ «срочно»; дальнейшие ответы тоже напомнят о нём (история)
    say(tx["bye"])
    return {"facts": facts, "survey": answers or None, "history": history}


def from_notebook(
    ns, what="Фото (Я2)", survey=True, lang="ru", ask=None, say=None, **kw
):
    """Ячейка Я5: берёт результат Я2 (r, Пол, Возраст, Беременность) и/или Я3 (lab, Пол_анализ, …) из
    переменных блокнота ns. Если профили Я2 и Я3 различаются — не смешивает их, а просит исправить."""
    say = say or (lambda text: print(text))
    use_photo, use_lab = "Фото" in what, "нализ" in what
    photo = ns.get("r") if use_photo else None
    blood = ns.get("lab") if use_lab else None
    if use_photo and photo is None:
        say("Нет результата фото: сначала запустите ячейку Я2.")
        return None
    if use_lab and blood is None:
        say("Нет анализа крови: сначала запустите ячейку Я3.")
        return None
    p2 = profile_from_form(ns.get("Пол"), ns.get("Возраст"), ns.get("Беременность"))
    p3 = profile_from_form(
        ns.get("Пол_анализ"), ns.get("Возраст_анализ"), ns.get("Беременность_анализ")
    )
    if use_photo and use_lab and p2 != p3:
        say(
            "Пол, возраст или беременность в Я2 и Я3 различаются: норма анализа посчитана для другого человека. "
            "Исправьте поля, перезапустите Я2 или Я3, затем Я5."
        )
        return None
    return chat(
        photo,
        blood,
        p2 if use_photo else p3,
        ask=ask,
        say=say,
        survey=survey,
        lang=lang,
        **kw,
    )
