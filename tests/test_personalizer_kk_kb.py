"""Казахская база знаний kb/kk.md (ИИ-перевод ru.md, 2026-11-18): та же база на двух языках и те же запреты.

Перевод — черновик: смысл и термины должны проверить носитель языка и врач. Тесты проверяют только то, что
проверяется кодом: те же темы и слоты, казахский текст, нет доз, препаратов, диагноза и выдуманных цифр,
срочные признаки и отказы срабатывают раньше базы.
"""

import re

import pytest

from anemia.personalizer import texts as T
from anemia.personalizer import texts_kk as TK
from anemia.personalizer.engine import ABOUT_RESULT, answer, kk_fill
from anemia.personalizer.facts import build_facts
from anemia.personalizer.kbtools import SAMPLE_CARDS
from anemia.personalizer.retrieve import SLOT, load_kb
from anemia.personalizer.router import KK_LETTERS, lang_of
from anemia.personalizer_eval import violations

RU = {fr.id: fr for fr in load_kb("ru")}
KK = {fr.id: fr for fr in load_kb("kk")}
CARDS = [(name, build_facts(p, lab, prof)) for name, p, lab, prof in SAMPLE_CARDS]

# казахские формы запретов (русские SAFETY их не видят): «сізде анемия бар», «темірді ішіңіз», «2 рет ішіңіз»
KK_DIAGNOSIS = re.compile(
    r"сізде\s+(?:\w+\s+){0,2}анемия\s+бар(?!\s*(?:ма|ме|екен|деп))", re.IGNORECASE
)
KK_PRESCRIBE = re.compile(
    r"(?:темір\w*|препарат\w*|таблетка\w*|дәрі\w*)\s+(?:ішіңіз|қабылдаңыз|егіңіз)(?!\s*деп)",
    re.IGNORECASE,
)
KK_DOSE = re.compile(r"\d+\s*(?:рет|таблетка|капсула|мг|мкг)\b", re.IGNORECASE)


def test_same_topics_and_slots_as_russian():
    assert set(KK) == set(RU)
    for fid, fr in KK.items():
        assert sorted(set(fr.slots)) == sorted(set(RU[fid].slots)), fid
        assert set(fr.slots) <= set(TK.SLOTS), fid
    assert set(TK.SLOTS) == set(T.SLOTS)


def test_every_topic_is_marked_as_ai_translation():
    for fid, fr in KK.items():
        assert "ИИ аудармасы" in fr.src, fid
        assert fr.examples, fid
        if "нужна проверка врача" in RU[fid].src:
            assert "дәрігер тексеруі керек" in fr.src, (
                fid
            )  # новые медицинские темы — с пометкой для врача


def test_static_text_has_no_hb_numbers_or_drugs():
    for fid, fr in KK.items():
        static = SLOT.sub("", fr.body)
        assert not re.search(r"\d{2,3}\s*(?:–\s*\d{2,3}\s*)?г/л", static), fid
        assert not KK_DOSE.search(static), fid


@pytest.mark.parametrize("name,facts", CARDS, ids=[c[0] for c in CARDS])
def test_filled_bodies_pass_guard(name, facts):
    for fid, fr in KK.items():
        if (
            facts.get("photo", {})
            and facts["photo"].get("status") != "ok"
            and fid in ABOUT_RESULT
        ):
            continue  # при непригодном снимке движок отвечает «переснимите»
        q = fr.examples[0]
        text = kk_fill(fr.body, facts, q)
        if not text:
            continue  # тема — одни факты, а их нет (анализа нет): движок даёт общее объяснение результата
        assert len(KK_LETTERS.findall(text)) >= 3, (fid, text)
        assert not violations(text, facts, q, "kk"), (
            fid,
            violations(text, facts, q, "kk"),
        )
        assert not KK_DIAGNOSIS.search(text), (fid, text)
        assert not KK_PRESCRIBE.search(text), (fid, text)
        assert not KK_DOSE.search(text), (fid, text)
        assert "{" not in text, (fid, text)


@pytest.mark.parametrize(
    "q,topic",
    [
        ("Қандай тағам жеу керек?", "nutrition"),
        ("каным аз не жесем болады", "nutrition"),  # без казахских букв, разговорное
        ("Қаназдық деген не?", "what_is_anemia"),
        ("Анемия емделе ме?", "treatment_course"),
        ("Ферритин деген не?", "ferritin"),
        ("Шай ишуге бола ма", "tea"),
        ("Етеккір көп келеді", "heavy_periods"),
        ("Темірді өзім ішсем бола ма", "self_iron"),
        ("мен жүктімін анемия қауіпті ме", "pregnancy"),
        ("қай дәрігерге бару керек", "which_doctor"),
    ],
)
def test_kazakh_questions_find_topic(q, topic):
    out = answer(q, build_facts(*SAMPLE_CARDS[0][1:]))
    assert out["lang"] == "kk" and out["sources"] == [topic], out
    assert len(KK_LETTERS.findall(out["text"])) >= 3


def test_rules_still_win_over_kazakh_kb():
    f = build_facts(*SAMPLE_CARDS[0][1:])
    assert answer("Қандай дәрі ішу керек?", f)["text"] == T.KK_DOSE
    assert answer("Темір дәрісін қанша ішу керек?", f)["text"] == T.KK_DOSE
    assert answer("Баламды тексеруге бола ма?", f)["mode"] == "child"
    assert answer("Кеудем қатты ауырады", f)["mode"] == "urgent"
    assert answer("Ауа райы қандай?", f)["mode"] == "out_of_scope"


def test_unknown_kazakh_question_says_dont_know_in_kazakh():
    out = answer("Тісімді емдетуге бола ма?", build_facts(*SAMPLE_CARDS[0][1:]))
    assert out["mode"] in ("dont_know", "out_of_scope") and out["lang"] == "kk"
    assert len(KK_LETTERS.findall(out["text"])) >= 3


def test_bad_photo_kazakh():
    f = build_facts(*SAMPLE_CARDS[2][1:])
    assert answer("Нәтижем нені білдіреді?", f)["text"] == T.KK_RETAKE
    general = answer("Қандай тағам жеу керек?", f)
    assert general["sources"] == ["nutrition"] and general["text"].endswith(
        TK.RETAKE_NOTE
    )


@pytest.mark.parametrize(
    "q",
    [
        "Можно ли пить чай, Ма?",
        "Что такое ферритин?",
        "Анемия лечится?",
        "Где купить, пе",
    ],
)
def test_russian_stays_russian(q):
    assert lang_of(q) == "ru"
