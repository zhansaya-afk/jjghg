"""Проверка базы знаний (Я6, scripts/check_kb.py) и строка «как найден ответ» (Я5)."""

from anemia.personalizer import answer, build_facts
from anemia.personalizer.kbtools import KB_FILE, check_kb, explain
from anemia.personalizer_eval import load_cases

F = build_facts({"status": "ok", "risk": "высокий", "hb_range": [92, 112], "tissues": ["conj"]}, None,
                {"sex": "F", "age": 30, "pregnant": False})  # fmt: skip


def run_check(path=None, **kw):
    said = []
    ok = check_kb(path, say=said.append, **kw)
    return ok, "\n".join(said)


def test_current_knowledge_base_passes():
    ok, out = run_check(exam=False)
    assert ok, out


def test_broken_knowledge_base_is_reported(tmp_path):
    held = next(
        c["question"]
        for c in load_cases(split="holdout")
        if c["expect_mode"] == "answer"
    )
    bad = KB_FILE.read_text(encoding="utf-8") + (
        "\n## Новая тема\nq: как поднять гемоглобин быстро\n\nТекст без разделителя.\n"
        "\n## dose_bad\nq: сколько пить\n---\nПейте сорбифер по 2 таблетки, норма 120 г/л. Обсудите с {doctor}. {oops}\n"
        f"\n## leak\nq: {held}\n---\nТекст.\n"
    )
    path = tmp_path / "ru.md"
    path.write_text(bad, encoding="utf-8")
    ok, out = run_check(path, exam=True)
    assert not ok
    for msg in ("имя темы", "нет строки ---", "неизвестные слоты", "почти совпадает с отложенным вопросом",
                "экзамен не запускался"):  # fmt: skip
        assert msg in out, (msg, out)
    fixed = bad.replace(" {oops}", "")
    path.write_text(fixed, encoding="utf-8")
    _, out = run_check(path, exam=False)
    assert (
        "цифры Hb" in out and "дательн" not in out and "{doctor} подставляется" in out
    )
    assert "текст темы dose_bad" in out  # препарат и доза пойманы стражем


def test_explain_tells_why():
    assert "правило: доза" in explain(answer("Сколько таблеток пить?", F))
    low = explain(answer("Пью чай после обеда, это плохо?", F))
    assert "< порога 0,40" in low and "строку q:" in low
    h = [
        {"role": "user", "text": "Какая норма гемоглобина?"},
        {"role": "assistant", "text": "…"},
    ]
    assert "искал вместе с прошлым вопросом" in explain(answer("А для мужчин?", F, h))
    assert "опасный признак" in explain(
        answer("Вчера был чёрный стул, железо не пью", F)
    )
