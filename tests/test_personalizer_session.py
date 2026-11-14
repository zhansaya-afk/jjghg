"""Ячейка Я5: диалог персонализатора на поддельном вводе и сам код ячейки из блокнота."""

import builtins
import json
from pathlib import Path

import pytest

from anemia.personalizer.session import chat, from_notebook, yes_no
from anemia.personalizer_eval import violations

NB = Path(__file__).resolve().parents[1] / "notebooks" / "colab_train.ipynb"
PHOTO = {  # как возвращает predict.analyze_photo
    "status": "ok",
    "stage": "ok",
    "risk": "высокий",
    "p_anemia": 0.81,
    "hb_range": (92.0, 112.0),
    "tissues": ["conj", "sclera"],
    "issues": [],
    "reasons": [],
}
BAD = {"status": "retake", "stage": "no_eye", "risk": None, "hb_range": None, "tissues": [],
       "issues": ["глаз не найден — снимите глаз с оттянутым нижним веком"]}  # fmt: skip
WOMAN = {"sex": "F", "age": 32, "pregnant": False}


def run(answers, *args, **kw):
    said, it = [], iter(answers)
    out = chat(*args, ask=lambda p: next(it, ""), say=said.append, **kw)
    return out, [s for s in said if s]


def test_yes_no():
    assert [yes_no(x) for x in ("Да", "нет", "иә", "жоқ", "", "не знаю")] == [
        True,
        False,
        True,
        False,
        None,
        None,
    ]


def test_full_conversation_uses_survey_and_stays_safe():
    out, said = run(
        ["нет", "да", "да", "нет", "нет", "Какие анализы сдать?", ""],
        PHOTO,
        None,
        WOMAN,
    )
    assert out["survey"] == {"red_flag": False, "weakness": True, "blood_loss": True, "meat_fish": False,
                             "family_thalassemia": False}  # fmt: skip
    assert "высокий" in said[0]  # сначала объяснение результата фото
    after = said[2]  # после опроса: подсказки по ответам + анализы
    assert (
        "месячные" in after and "B12" in after and "общий анализ крови" in after.lower()
    )
    assert any("ферритин" in s for s in said[3:])
    for s in said:
        assert not violations(s, out["facts"]), s


def test_red_flag_in_survey_stops_with_urgent():
    out, said = run(["да"], PHOTO, None, WOMAN)
    assert out["survey"] == {"red_flag": True}  # остальное не спрашивается
    assert any("103" in s for s in said) and "сначала обратитесь" in said[-1]


def test_child_refused_without_survey():
    child = {"status": "unreliable", "stage": "age", "risk": None, "issues": []}
    out, said = run([], child, None, {"sex": "F", "age": 3})
    assert out["survey"] is None and "5 лет" in said[0]


def test_bad_photo_asks_retake_with_reason():
    _, said = run(["", "", "", "", ""], BAD, None, WOMAN)
    assert said[0].startswith("Снимок не подошёл (глаз не найден)")


def test_lab_only_and_kazakh():
    lab = {"hb_g_l": 104}
    out, said = run(["жоқ", "иә", "жоқ", "иә", "жоқ", ""], None, lab, WOMAN, lang="kk")
    assert "104" in said[0]
    assert out["survey"]["weakness"] is True


def test_no_interactive_input_ends_cleanly():
    def eof(prompt):
        raise EOFError

    said = []
    out = chat(PHOTO, None, WOMAN, ask=eof, say=said.append, survey=False)
    assert out["history"] and "диагноз" in said[-1]


def test_from_notebook_refuses_mixed_profiles():
    ns = {"r": PHOTO, "профиль_Я2": ("Женский", 32, "Нет"),
          "lab": {"hb_g_l": 104}, "Пол_анализ": "Мужской", "Возраст_анализ": 32, "Беременность_анализ": "Нет"}  # fmt: skip
    said = []
    assert from_notebook(ns, "Фото и анализ", say=said.append) is None
    assert "различаются" in said[0]
    said = []
    assert from_notebook({}, "Фото (Я2)", say=said.append) is None and "Я2" in said[0]


def _cell(title):
    nb = json.loads(NB.read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    cells = [c for c in code if title in "".join(c["source"])]
    # Я5 и Я6 — последние ячейки блокнота, остальные не тронуты
    assert len(cells) == 1 and [code[-2], code[-1]] == [
        c
        for c in code
        if "Ячейка Я5" in "".join(c["source"]) or "Ячейка Я6" in "".join(c["source"])
    ]
    return "".join(cells[0]["source"])


def _ya5_source():
    return _cell("Ячейка Я5")


def test_ya5_cell_runs_on_ya2_variables(monkeypatch, capsys):
    answers = iter(["нет", "нет", "нет", "да", "нет", "Что такое ферритин?", ""])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers, ""))
    ns = {"r": PHOTO, "профиль_Я2": ("Женский", 32, "Нет")}
    exec(compile(_ya5_source(), "Я5", "exec"), ns)  # noqa: S102 — код ячейки блокнота
    out = capsys.readouterr().out
    assert "По фото риск анемии высокий" in out and "Ферритин" in out


def test_ya5_keeps_ya2_profile_after_ya2s(monkeypatch, capsys):
    """Я2 → Я2с → Я5: Я2с перезаписывает Пол / Возраст / Беременность, Я5 берёт профиль из Я2."""
    monkeypatch.setattr(builtins, "input", lambda prompt="": "")
    ya2 = _cell("Ячейка Я2 —").splitlines()
    run_at = next(i for i, s in enumerate(ya2) if s.startswith("r = analyze_photo("))
    keep = next(s for s in ya2 if s.startswith("профиль_Я2 ="))
    assert ya2.index(keep) > run_at  # профиль запоминается вместе с результатом Я2
    ns = {"r": PHOTO, "Пол": "Мужской", "Возраст": 45, "Беременность": "Нет"}  # поля Я2
    exec(keep, ns)  # noqa: S102 — строка ячейки Я2
    ns.update({"Пол": "Женский", "Возраст": 4, "Беременность": "Да"})  # затем Я2с с другими полями
    exec(compile(_ya5_source(), "Я5", "exec"), ns)  # noqa: S102 — код ячейки блокнота
    out = capsys.readouterr().out
    assert ns["_"]["facts"]["profile"] == {"sex": "M", "age": 45, "pregnant": False}
    assert "По фото риск анемии высокий" in out and "младше 5" not in out and "беремен" not in out


@pytest.mark.parametrize("what", ["Анализ крови (Я3)", "Фото и анализ"])
def test_ya5_cell_without_ya3_says_what_to_run(what, capsys):
    src = _ya5_source().replace(
        "Что_объяснить = 'Фото (Я2)'", f"Что_объяснить = {what!r}"
    )
    exec(  # noqa: S102 — код ячейки блокнота
        compile(src, "Я5", "exec"),
        {"r": PHOTO, "профиль_Я2": ("Женский", 32, "Нет")},
    )
    assert "Я3" in capsys.readouterr().out


def test_chat_remembers_said_facts_until_the_end():
    out, said = run(
        ["", "", "", "", "", "Я вегетарианка", "Какие анализы сдать?", ""],
        PHOTO,
        None,
        WOMAN,
    )
    note = [s for s in said if s.startswith("Железо есть")]
    assert (
        note and "Учту до конца этого разговора: вы не едите мясо или рыбу." in note[0]
    )
    assert (
        "B12" in said[-2]
    )  # анализы с учётом сказанного; карточка разговора не меняется
    assert out["facts"]["survey"].get("meat_fish") is None


def test_ya5_cell_shows_how_answer_was_found(monkeypatch, capsys):
    answers = iter(["Пью чай после обеда, это плохо?", ""])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers, ""))
    src = (
        _ya5_source()
        .replace("Опрос = True", "Опрос = False")
        .replace(
            "Показывать_как_найден_ответ = False", "Показывать_как_найден_ответ = True"
        )
    )
    ns = {"r": PHOTO, "профиль_Я2": ("Женский", 32, "Нет")}
    exec(compile(src, "Я5", "exec"), ns)  # noqa: S102 — код ячейки блокнота
    out = capsys.readouterr().out
    assert "🔎 поиск: тема" in out and "порога" in out


def test_ya6_cell_checks_the_knowledge_base(capsys):
    src = _cell("Ячейка Я6").replace(
        "Мои_вопросы = ''", "Мои_вопросы = 'можно ли кофе | сколько таблеток пить'"
    )
    exec(compile(src, "Я6", "exec"), {})  # noqa: S102 — код ячейки блокнота
    out = capsys.readouterr().out
    assert "Итог: ✅ ошибок нет" in out and "экзамен dev: пройдено 58/58" in out
    assert "тема tea" in out and "правило: доза" in out


def test_ya5_cell_uses_neural_net_from_cell_n(monkeypatch, capsys):
    answers = iter(["Со временем станет только хуже?", ""])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers, ""))
    src = (
        _ya5_source()
        .replace("Опрос = True", "Опрос = False")
        .replace(
            "Нейросеть_понимает_вопрос = False", "Нейросеть_понимает_вопрос = True"
        )
        .replace(
            "Показывать_как_найден_ответ = False", "Показывать_как_найден_ответ = True"
        )
    )
    ns = {"r": PHOTO, "профиль_Я2": ("Женский", 32, "Нет")}
    exec(compile(src, "Я5", "exec"), dict(ns))  # noqa: S102 — без ячейки Н
    assert "запустите ячейку Н" in capsys.readouterr().out
    answers = iter(["Со временем станет только хуже?", ""])
    ns["LLM"] = lambda messages, **kw: "can_worsen"
    exec(compile(src, "Я5", "exec"), ns)  # noqa: S102 — код ячейки блокнота
    out = capsys.readouterr().out
    assert "нейросеть поняла вопрос: тема can_worsen" in out
