"""Проверка режима «нейросеть понимает вопрос» (ячейка Я7): правила против правил + нейросеть.

compare(llm) прогоняет на одной и той же карточке (фото: высокий риск, Ж 32):
- P — 30 вопросов пользователя другими словами (перефразы; написаны до прогона, под результат не правятся);
- U — сами 30 вопросов пользователя 2026-11-07;
- A — набор замера поиска (128 вопросов, его можно смотреть);
- экзамен dev и holdout: опасные ответы, пропущенные тревожные признаки, лишние «срочно».
Нейросеть вызывается только для вопросов, которые правила не нашли; ответ — всегда текст базы.
"""

import json
import time
from functools import partial
from pathlib import Path

from .. import personalizer_eval as pe
from .engine import answer
from .facts import build_facts

PARAPHRASES = Path(pe.CASES_PATH).parent / "paraphrases_user30.json"
CARD = (
    {"status": "ok", "risk": "высокий", "hb_range": [92, 112], "tissues": ["conj"]},
    None,
    {"sex": "F", "age": 32, "pregnant": False},
)


def _grade(out, ok):
    silent = out["mode"] in ("dont_know", "out_of_scope")
    if ok is None:
        return "ok" if silent else "wrong"
    if set(out.get("sources") or ()) & set(ok):
        return "ok"
    return "idk" if silent else "wrong"


def _rows(fn, questions, facts):
    rows = []
    for x in questions:
        out = fn(x["q"], facts)
        rows.append(
            {
                "q": x["q"],
                "ok": x["ok"],
                "grade": _grade(out, x["ok"]),
                "got": ",".join(out.get("sources") or ()) or out["mode"],
                "engine": out.get("engine", "rules"),
                "guard": bool(out.get("guard")),
            }
        )
    return rows


def _count(rows):
    c = {g: sum(r["grade"] == g for r in rows) for g in ("ok", "wrong", "idk")}
    return f"верно {c['ok']}/{len(rows)}, не та тема {c['wrong']}, «не знаю» {c['idk']}"


def compare(llm, say=print, details=True, exam=True):
    """→ dict с итогами; печатает таблицу «правила / правила + нейросеть» и разбор P и U по вопросам."""
    facts = build_facts(*CARD)
    rules = answer
    smart = partial(answer, engine="understand", llm=llm)
    sets = json.loads(PARAPHRASES.read_text(encoding="utf-8"))["questions"]
    groups = {
        "P перефразы (30)": [x for x in sets if x["split"] == "P"],
        "U вопросы пользователя (30)": [x for x in sets if x["split"] == "U"],
        "A замер поиска (128)": pe.load_search_questions(split="A"),
    }
    res, t0 = {}, time.time()
    say("Набор | только правила | правила + нейросеть")
    for name, qs in groups.items():
        a, b = _rows(rules, qs, facts), _rows(smart, qs, facts)
        res[name] = (a, b)
        say(f"{name} | {_count(a)} | {_count(b)}")
    if exam:
        for split in ("dev", "holdout"):
            cases = pe.load_cases(split=split)
            s = pe.summary(pe.run(smart, cases), cases)
            res[f"exam_{split}"] = s
            say(
                f"Экзамен {split} с нейросетью | пройдено {s['passed']['k']}/{s['passed']['n']}, "
                f"опасных {s['unsafe']['k']}, пропущено тревожных признаков {s['red_flag_missed']}, "
                f"лишних «срочно» {s['false_urgent']}"
            )
    say(f"Время: {time.time() - t0:.0f} с")
    if details:
        for name in list(groups)[:2]:
            say(f"\n{name}: по каждому вопросу (правила → с нейросетью)")
            for i, (a, b) in enumerate(zip(*res[name]), 1):
                mark = {"ok": "✅", "wrong": "❌", "idk": "…"}
                say(
                    f"{i:2d}. {mark[a['grade']]}→{mark[b['grade']]} {b['q']} — "
                    f"{a['got']} → {b['got']}{' (нейросеть)' if b['engine'] == 'understand' else ''}"
                )
        say(
            "✅ верно, ❌ не та тема, … «не знаю». Ответ всегда — проверенный текст базы."
        )
    return res
