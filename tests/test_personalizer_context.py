"""Память в пределах одного разговора: уточнения («А для мужчин?») и факты о себе из чата."""

import pytest

from anemia.personalizer import answer, build_facts
from anemia.personalizer import texts as T
from anemia.personalizer.context import (
    context_query,
    dialog_facts,
    is_follow_up,
    new_facts_note,
    said_facts,
)
from anemia.personalizer_eval import violations

PHOTO = {
    "status": "ok",
    "risk": "высокий",
    "hb_range": [92, 112],
    "tissues": ["conj", "sclera"],
}
WOMAN = {"sex": "F", "age": 32, "pregnant": False}


def talk(questions, result=PHOTO, lab=None, profile=WOMAN):
    """Разговор без опроса → ответы по очереди (каждый проверен стражем)."""
    f, history, outs = build_facts(result, lab, profile), [], []
    for q in questions:
        o = answer(q, f, history)
        assert not violations(o["text"], o["facts"], o["context"] or q), (q, o["text"])
        history += [
            {"role": "user", "text": q},
            {"role": "assistant", "text": o["text"]},
        ]
        outs.append(o)
    return outs


@pytest.mark.parametrize(
    "q,want",
    [
        ("А для мужчин?", True),
        ("А при беременности?", True),
        ("А если 105?", True),
        ("Почему?", True),
        ("Мужчинам?", True),
        ("А если я донор?", False),  # своя тема — ищется сама
        ("А это точно?", False),
        ("Что делать?", False),
        ("Привет", False),
        ("", False),
    ],
)
def test_is_follow_up(q, want):
    assert is_follow_up(q) is want


def test_context_query_uses_last_own_question():
    h = [
        {"role": "user", "text": "Какая норма гемоглобина?"},
        {"role": "assistant", "text": "…"},
        {"role": "user", "text": "А для мужчин?"},
        {"role": "assistant", "text": "…"},
    ]
    assert (
        context_query("А при беременности?", h)
        == "Какая норма гемоглобина? А при беременности?"
    )
    assert context_query("Что такое ферритин?", h) == "Что такое ферритин?"
    assert context_query("А для мужчин?", []) == "А для мужчин?"


@pytest.mark.parametrize(
    "text,want",
    [
        ("Я вегетарианка", {"meat_fish": False}),
        ("я не ем мясо и рыбу", {"meat_fish": False}),
        ("Мясо я не ем, что делать?", {"meat_fish": False}),
        ("я ем мясо каждый день", {"meat_fish": True}),
        ("Я не вегетарианка", {}),
        ("Что есть веганам?", {}),  # вопрос о других — не факт о себе
        ("Я донор, сдаю кровь каждые 3 месяца", {"donor": True}),
        ("А если я донор?", {}),
        ("Я сдаю кровь на анализ завтра", {}),
        ("Я не донор", {}),
        ("У мамы талассемия", {"family_thalassemia": True}),
        ("У мамы нет талассемии", {}),
        ("У меня обильные месячные", {"heavy_periods": True}),
        ("У меня не обильные месячные", {}),
        ("У меня хроническая болезнь почек", {"chronic_disease": "почки"}),
        ("У меня нет болезни почек", {}),
        ("Анемия бывает при болезни почек?", {}),
        ("У меня язва желудка", {"chronic_disease": "ЖКТ"}),
    ],
)
def test_said_facts(text, want):
    assert said_facts([text]) == want


def test_dialog_facts_override_survey_without_changing_card():
    f = build_facts(PHOTO, None, WOMAN, {"meat_fish": True})
    h = [{"role": "user", "text": "Я перешла на веганство"}]
    g = dialog_facts(f, h)
    assert g["survey"]["meat_fish"] is False and f["survey"]["meat_fish"] is True
    assert (
        g["profile"] == f["profile"] and g["photo"] == f["photo"]
    )  # риск и профиль не меняются
    assert (
        new_facts_note(f, g)
        == "Учту до конца этого разговора: вы не едите мясо или рыбу."
    )
    assert dialog_facts(f, []) is f


def test_follow_up_about_other_sex_norm():
    first, second, third = talk(
        ["Какая норма гемоглобина?", "А для мужчин?", "А при беременности?"]
    )
    assert "130" not in first["text"]
    assert (
        second["context"]
        and "130 г/л" in second["text"]
        and "120 г/л" in second["text"]
    )
    assert "110 г/л" in third["text"]


def test_follow_up_number_compared_with_other_sex():
    _, o = talk(["У меня 105, это анемия?", "А для мужчин?"], result=None)
    assert (
        "Для мужчин порог ВОЗ выше — 130 г/л. 105 г/л ниже этого порога." in o["text"]
    )


def test_how_much_after_iron_question_is_a_dose_refusal():
    _, o = talk(["Можно ли принимать железо?", "А сколько?"])
    assert o["mode"] == "answer" and o["text"] == T.dose(o["facts"])


def test_follow_up_without_question_asks_to_clarify():
    _, o = talk(["Привет", "А для мужчин?"])
    assert o["mode"] == "dont_know" and o["text"] == T.CLARIFY


def test_red_flag_in_follow_up_stays_urgent():
    _, o = talk(["Что есть при анемии?", "А если стул чёрный, как дёготь?"])
    assert o["mode"] == "urgent"


def test_said_facts_change_tests_and_hints():
    *_, tests = talk(
        ["Что делать дальше?", "У мамы талассемия", "Какие анализы сдать?"]
    )
    assert "электрофорез гемоглобина" in tests["text"]
    *_, tests = talk(["Я вегетарианка", "Какие анализы сдать?"])
    assert "B12" in tests["text"]
    (tests,) = talk(["Я не вегетарианка. Какие анализы сдать?"])
    assert "B12" not in tests["text"]
    *_, nxt = talk(["Я донор", "Что делать дальше?"])
    assert "сдаче крови" in nxt["text"]


@pytest.mark.parametrize(
    "q,source",
    [
        ("Подруга пьёт железо и ей помогло, можно я тоже?", "self_iron"),
        ("Можно ли сдать анализ после еды?", "lab_prep"),
        ("Анемия бывает от стресса?", "stress"),
        ("Мне 16 лет, это считается?", "age_scope"),
        ("Когда пересдать анализ?", "how_often"),
        ("Я беременна, опасна ли анемия для малыша?", "pregnancy"),
        ("сколько было с анемией", "dataset"),
        ("через сколько риск станет низким", "how_often"),
    ],
)
def test_found_weak_answers(q, source):
    """Вопросы, на которые раньше был слабый ответ (их нашли при проверке в Colab), — не кейсы экзамена."""
    o = answer(q, build_facts(PHOTO, None, WOMAN))
    assert o["sources"] == [source] and not violations(o["text"], o["facts"], q)


def test_accuracy_by_group_and_past_and_mmol():
    f = build_facts(PHOTO, None, WOMAN)
    assert (
        "Отдельно для женщин точность не считали"
        in answer("Насколько точно для женщин?", f)["text"]
    )
    assert answer("Сравни с прошлой неделей", f)["mode"] == "dont_know"
    o = answer(
        "У меня гемоглобин 6,2 ммоль/л, это нормально?", build_facts(None, None, WOMAN)
    )
    assert "100 г/л" in o["text"] and "глюкоз" in o["text"] and not o["guard"]
