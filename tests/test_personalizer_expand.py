"""Расширение Personalizer 2026-11-15: новые темы, вопросы без слова «анемия», «это» из прошлого вопроса,
нейросеть выбирает тему только из 3 лучших тем поиска."""

import pytest

from anemia.personalizer import answer, build_facts
from anemia.personalizer import engine as E
from anemia.personalizer.retrieve import default_retriever, load_kb
from anemia.personalizer_eval import violations

PHOTO = {"status": "ok", "risk": "высокий", "hb_range": [92, 112], "tissues": ["conj"]}
WOMAN = {"sex": "F", "age": 32, "pregnant": False}
NEW_TOPICS = (
    "prevention",
    "living_with",
    "anemia_types",
    "cbc_basics",
    "meal_example",
    "heavy_periods",
)


def facts():
    return build_facts(PHOTO, None, WOMAN)


def talk(questions, **kw):
    f, history, outs = facts(), [], []
    for q in questions:
        o = answer(q, f, history, **kw)
        assert not violations(o["text"], o["facts"], o["context"] or q), (q, o["text"])
        history += [
            {"role": "user", "text": q},
            {"role": "assistant", "text": o["text"]},
        ]
        outs.append(o)
    return outs


def test_new_topics_marked_for_doctor_review():
    kb = {fr.id: fr for fr in load_kb("ru")}
    for t in (*NEW_TOPICS, "tea"):
        assert "нужна проверка врача" in kb[t].src, t


@pytest.mark.parametrize(
    "q,want",
    [
        ("Можно ли предотвратить анемию?", "prevention"),  # вопрос пользователя 14.11
        ("как справляться с анемией", "living_with"),  # вопрос пользователя 14.11
        ("Какие типы анемии существуют?", "anemia_types"),
        ("Что означает гематокрит в анализе?", "cbc_basics"),
        ("Распиши меню на день с железом", "meal_example"),
        ("Месячные очень обильные, гемоглобин может упасть?", "heavy_periods"),
        ("Молоко и творог мешают железу?", "tea"),
    ],
)
def test_new_topics_found(q, want):
    o = answer(q, facts())
    assert o["sources"] == [want], (q, o["trace"])
    assert not violations(o["text"], o["facts"], q)


@pytest.mark.parametrize(
    "q,want",
    [
        (
            "На парах выдыхаюсь",
            "fatigue",
        ),  # вопрос пользователя 14.11: раньше «не моя тема»
        ("Можно ли это как-то предотвратить", "prevention"),  # и без прошлого вопроса
        ("Как часто программа ошибается?", "accuracy"),
    ],
)
def test_questions_without_word_anemia_pass(q, want):
    assert answer(q, facts())["sources"] == [want]


@pytest.mark.parametrize(
    "q",
    [
        "Как сдать ЕНТ на высокий балл?",
        "Что приготовить на день рождения?",
        "Как выучить химию за неделю?",
        "Какой ноутбук купить?",
    ],
)
def test_off_topic_still_refused(q):
    assert answer(q, facts())["mode"] in ("out_of_scope", "dont_know")


def test_refers_back_uses_previous_question():
    o = talk(["Что такое ферритин?", "Зачем его сдавать?"])[1]
    assert o["sources"] == ["ferritin"]
    assert any("прошлом вопросе" in n for n in o["trace"]["notes"])


def test_refers_back_keeps_own_topic():
    # «Это опасно?» — своя тема (serious), прошлый вопрос о ферритине её не перебивает
    assert talk(["Что такое ферритин?", "Это опасно?"])[1]["sources"] == ["serious"]
    assert talk(["Можно ли пить чай с едой?", "Пью чай после обеда, это плохо?"])[1][
        "sources"
    ] == ["tea"]


def test_refers_back_not_after_off_topic():
    assert talk(["Как выучить химию?", "Это сложно?"])[1]["mode"] in (
        "out_of_scope",
        "dont_know",
    )


def test_refers_back_never_skips_safety_rules():
    outs = talk(["Что такое анемия?", "Это значит у меня анемия?"])
    assert outs[1]["why"] == "diagnosis"
    outs = talk(["Что такое анемия?", "Сколько мг этого пить?"])
    assert outs[1]["why"] == "dose"
    outs = talk(["Что такое анемия?", "От этого потеряла сознание"])
    assert outs[1]["mode"] == "urgent"


def _picker(reply, seen):
    def llm(messages, **kw):
        seen.append(messages[0]["content"])
        return reply

    return llm


def _catalog_ids(system):
    lines = system.split("Темы:\n", 1)[1].splitlines()
    return [ln.split(":", 1)[0] for ln in lines if ln.strip()]


def test_llm_sees_only_top3_search_topics():
    q = "Если запустить, что будет?"
    assert answer(q, facts())["mode"] in ("out_of_scope", "dont_know")
    top = [fr.id for fr, _ in default_retriever("ru").search(q, k=E.LLM_TOP_K)]
    seen = []
    r = answer(q, facts(), engine="understand", llm=_picker("can_worsen", seen))
    assert _catalog_ids(seen[0]) == top and len(top) == 3
    assert "can_worsen" in top and r["sources"] == ["can_worsen"]


def test_llm_topic_outside_top3_rejected():
    q = "Если запустить, что будет?"
    top = {fr.id for fr, _ in default_retriever("ru").search(q, k=E.LLM_TOP_K)}
    outside = next(fr.id for fr in load_kb("ru") if fr.id not in top)
    r = answer(q, facts(), engine="understand", llm=_picker(outside, []))
    assert r["engine"] == "rules" and r["mode"] in ("out_of_scope", "dont_know")
