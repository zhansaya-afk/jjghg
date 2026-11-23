"""Казахский: проверка 2026-11-23 — пропущенные красные флаги, дети младше 5 лет, возраст ≠ Hb.
Только распознавание; тексты ответов не менялись. Фразы — черновик, их должен проверить носитель языка."""

import pytest

from anemia.personalizer import build_facts, rules_answer
from anemia.personalizer.facts import hb_from_question

F_OK = {"status": "ok", "risk": "средний", "p_anemia": 0.5, "tissue": "конъюнктива"}


def facts():
    return build_facts(F_OK, None, {"sex": "F", "age": 30, "pregnant": None})


@pytest.mark.parametrize(
    "q,why",
    [
        ("Жүрегім шаншиды", "chest"),
        ("Кеуде тұсым ауырып тұр", "chest"),
        ("Жүрек тұсым шаншып ауырады", "chest"),
        ("Есімнен айырылдым", "faint"),
        ("Құсығымда қан бар", "vomit"),
        ("Етеккірім өте көп, тоқтамай жатыр", "bleeding"),
        ("Бәрінен шаршадым, өмірдің мәні жоқ", "crisis"),
    ],
)
def test_kk_red_flags_found(q, why):
    r = rules_answer(q, facts())
    assert (r["mode"], r["why"]) == ("urgent", why)


@pytest.mark.parametrize(
    "q",
    [
        "Жүрек ауруы анемияға әсер ете ме?",  # болезнь сердца как тема — не срочно
        "Анемия жүрекке әсер ете ме?",
        "Құс еті темірге бай ма?",
        "Етеккір кезінде қан кетеді, бұл қалыпты ма?",
        "Тыныс алу жаттығулары көмектесе ме?",
    ],
)
def test_kk_no_false_urgent(q):
    assert rules_answer(q, facts())["mode"] != "urgent"


@pytest.mark.parametrize(
    "q",
    [
        "Балам 3 жаста, оны тексеруге бола ма?",
        "3 жасар баланың суретін түсірсем бола ма?",
        "Ұлым 2 жаста, бозарып жүр",
        "Сәбиімнің гемоглобині қанша болу керек?",
    ],
)
def test_kk_child_under_5_refused(q):
    assert rules_answer(q, facts())["mode"] == "child"


@pytest.mark.parametrize("q", ["Балам 7 жаста, анемия бола ма?", "Ұлым 10 жаста, бозарып жүр"])
def test_kk_older_child_not_refused(q):
    assert rules_answer(q, facts())["mode"] != "child"


@pytest.mark.parametrize("q", ["Ұлым 10 жаста", "8 айлық сәби", "2 апта бұрын", "70 пайыз деген не", "5 жыл"])
def test_kk_age_and_time_not_hb(q):
    assert hb_from_question(q, facts()) is None


def test_hb_with_kk_time_still_read():
    assert hb_from_question("Гемоглобин 95, 2 ай бұрын тапсырдым", facts()) == (95, "г/л")


@pytest.mark.parametrize(
    "q,hb",
    [
        ("гемоглобин 95 айналасында", 95),
        ("Гемоглобин 95, айтыңызшы", 95),
        ("гемоглобин 105 жылдам түсті", 105),
        ("Hb 98 рет-рет өлшедім", 98),
    ],
)
def test_kk_words_starting_like_units_keep_hb(q, hb):
    assert hb_from_question(q, facts()) == (hb, "г/л")


@pytest.mark.parametrize(
    "q",
    [
        "Сәбіз гемоглобинді көтере ме?",
        "Сабина гемоглобині 110",
        "Балама 3 ай бойы темір беріп жүрмін",
        "Қызым 1 айдан бері темір ішеді",
    ],
)
def test_kk_not_child_false_refusal(q):
    assert rules_answer(q, facts())["mode"] != "child"


def test_kk_infant_months_refused():
    assert rules_answer("Балам 8 айлық, бозарып жүр", facts())["mode"] == "child"
