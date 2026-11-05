"""Инструменты для тех, кто дополняет базу знаний (kb/ru.md) без программирования.

explain(out) — «как найден ответ» одной-двумя строками (галочка в Я5).
check_kb()  — ячейка Я6 и scripts/check_kb.py: формат файла, запреты в текстах, копии отложенных вопросов
              экзамена, экзамен целиком и «какую тему находит мой вопрос». Инструкция — kb/README.md.
"""

import re
from difflib import SequenceMatcher
from pathlib import Path

from . import texts as T
from .engine import ABOUT_RESULT, THRESHOLD, answer, fill
from .facts import build_facts
from .retrieve import KB_DIR, SLOT, default_retriever, load_kb, parse_kb

KB_FILE = KB_DIR / "ru.md"
RULES = {
    "child": "ребёнок младше 5 лет → отказ",
    "greeting": "приветствие",
    "thanks": "благодарность",
    "about": "вопрос о самом помощнике",
    "price": "вопрос о цене",
    "past": "прошлые результаты (их у помощника нет)",
    "certificate": "справка",
    "dose": "доза → дозы не называем",
    "drug": "выбор препарата → препараты не называем",
    "diagnosis": "просьба поставить диагноз",
    "language": "не русский и не казахский язык",
    "out_of_scope": "в вопросе нет слов про анемию, анализы, питание или самочувствие → «не моя тема»",
}
# карточки фактов, на которых проверяется каждый текст базы (подставляются в {слоты})
SAMPLE_CARDS = (
    ("фото: высокий риск, Ж 30", {"status": "ok", "risk": "высокий", "hb_range": [92, 112], "tissues": ["conj"]},
     None, {"sex": "F", "age": 30, "pregnant": False}),
    ("фото: низкий риск, М 45", {"status": "ok", "risk": "низкий", "hb_range": [130, 150], "tissues": ["conj", "sclera"]},
     None, {"sex": "M", "age": 45, "pregnant": False}),
    ("фото не подошло, Ж 25", {"status": "retake", "stage": "quality", "reason": "снимок размыт"},
     None, {"sex": "F", "age": 25, "pregnant": False}),
    ("анализ 95 г/л, беременна", None, {"hb_g_l": 95}, {"sex": "F", "age": 28, "pregnant": True}),
    ("фото: средний + анализ 125 г/л, М 60", {"status": "ok", "risk": "средний", "tissues": ["conj"]},
     {"hb_g_l": 125}, {"sex": "M", "age": 60, "pregnant": False}),
)  # fmt: skip
DEMO = SAMPLE_CARDS[0]
_DOCTOR_CASE = re.compile(r"(?:\bс|принимает|назначает|решает|определяет)\s+\{doctor\}")
_STATIC_HB = re.compile(r"\d{2,3}\s*(?:–\s*\d{2,3}\s*)?г/л")
_ID = re.compile(r"^[a-z][a-z0-9_]*$")


def _num(x):
    return f"{x:.2f}".replace(".", ",")


def explain(out):
    """Ответ engine.answer → «🔎 как найден ответ» (тема, сходство, правило, уточнение)."""
    tr = out.get("trace") or {}
    parts = []
    if out.get("context"):
        parts.append(f"искал вместе с прошлым вопросом: «{out['context']}»")
    rule = tr.get("rule")
    if out.get("mode") == "urgent":
        parts.append(
            f"правило: опасный признак ({rule}) → «срочно к врачу», без поиска по базе"
        )
    elif rule:
        parts.append(f"правило: {RULES.get(rule, rule)}, без поиска по базе")
    hits = tr.get("search") or []
    if hits:
        best, score = hits[0]
        near = ", ".join(f"{i} {_num(s)}" for i, s in hits[1:])
        verdict = "≥" if score >= THRESHOLD else "<"
        parts.append(
            f"поиск: тема {best}, сходство {_num(score)} {verdict} порога {_num(THRESHOLD)}"
            + (f"; рядом: {near}" if near else "")
        )
        if score < THRESHOLD and out.get("mode") == "dont_know":
            parts.append(
                "ни одна тема не набрала порог → «не знаю». Если вопрос подходит к одной из тем, "
                "добавьте свою формулировку в её строку q: (kb/ru.md)"
            )
    parts += tr.get("notes") or []
    if out.get("sources"):
        parts.append(f"ответ из темы: {', '.join(out['sources'])}")
    if out.get("guard"):
        parts.append(f"страж заменил ответ безопасным: {', '.join(out['guard'])}")
    return "🔎 " + "; ".join(parts) + "."


def format_problems(text):
    """Ошибки формата kb/ru.md: [(номер строки, тема, что не так)]."""
    out, seen = [], {}
    starts = [m.start() for m in re.finditer(r"^## ", text, flags=re.MULTILINE)]
    for k, a in enumerate(starts):
        block = text[a : starts[k + 1] if k + 1 < len(starts) else len(text)]
        line = text.count("\n", 0, a) + 1
        fid = block[3:].splitlines()[0].strip()
        if not _ID.match(fid):
            out.append((line, fid, "имя темы — латиница, цифры и _, например my_topic"))
        if fid in seen:
            out.append(
                (line, fid, f"тема с таким именем уже есть (строка {seen[fid]})")
            )
        seen.setdefault(fid, line)
        head, sep, body = block.partition("\n---\n")
        if not sep:
            out.append((line, fid, "нет строки --- перед текстом ответа"))
        q = [ln for ln in head.splitlines()[1:] if ln.startswith("q:")]
        if not q or not [e for e in q[0][2:].split("|") if e.strip()]:
            out.append((line, fid, "нет строки q: с примерами вопросов через |"))
        for ln in head.splitlines()[1:]:
            if ln.strip() and not re.match(r"^(?:q|src|mode):", ln):
                out.append(
                    (line, fid, f"непонятная строка до ---: «{ln.strip()[:40]}»")
                )
        if sep and not body.strip():
            out.append((line, fid, "пустой текст ответа после ---"))
        bad = sorted(set(SLOT.findall(body)) - set(T.SLOTS))
        if bad:
            out.append(
                (line, fid, f"неизвестные слоты {bad}; доступные: {sorted(T.SLOTS)}")
            )
    return out


def text_problems(fragments):
    """Запреты в текстах ответов: цифры Hb без слота, падеж {doctor}, нарушения стража на 5 карточках."""
    from ..personalizer_eval import violations

    out = []
    for fr in fragments:
        static = SLOT.sub("", fr.body)
        if _STATIC_HB.search(static):
            out.append(
                (
                    fr.id,
                    "цифры Hb «… г/л» прямо в тексте: нормы и Hb — только через {norm_line}, {lab_line}",
                )
            )
        if _DOCTOR_CASE.search(fr.body):
            out.append(
                (
                    fr.id,
                    "{doctor} подставляется как «врачу-терапевту»: пишите «покажите {doctor}», «к {doctor}»",
                )
            )
        if set(fr.slots) - set(T.SLOTS):
            continue  # уже в ошибках формата
        for name, photo, lab, profile in SAMPLE_CARDS:
            facts = build_facts(photo, lab, profile)
            if photo and photo["status"] != "ok" and not lab and fr.id in ABOUT_RESULT:
                continue  # при непригодном снимке на эти темы движок отвечает «переснимите», текст не показывается
            q = fr.examples[0] if fr.examples else ""
            bad = violations(fill(fr.body, facts, q), facts, q)
            if bad:
                out.append(
                    (fr.id, f"{name}: {', '.join(f'{k} {v}' for k, v in bad.items())}")
                )
    return out


def example_problems(fragments, holdout_questions, dev_questions=()):
    """Формулировки q:, которые совпадают с отложенными вопросами экзамена (ошибка) или повторяются (предупреждение).
    Дословные вопросы dev разрешены: dev — для разработки (как в tests/test_personalizer.py)."""

    def norm(s):
        return re.sub(r"[^\w]+", " ", s.lower().replace("ё", "е")).strip()

    errors, warnings, owner = [], [], {}
    held = [norm(h) for h in holdout_questions]
    dev = {norm(d) for d in dev_questions}
    for fr in fragments:
        for e in fr.examples:
            n = norm(e)
            if n in owner and owner[n] != fr.id:
                warnings.append(
                    (
                        fr.id,
                        f"«{e}» есть и в теме {owner[n]}: поиск выберет одну из них случайно",
                    )
                )
            owner.setdefault(n, fr.id)
            close = max(
                held, key=lambda h: SequenceMatcher(None, n, h).ratio(), default=""
            )
            if n in held or (
                n not in dev
                and close
                and SequenceMatcher(None, n, close).ratio() > 0.82
            ):
                errors.append(
                    (
                        fr.id,
                        f"«{e}» почти совпадает с отложенным вопросом экзамена — уберите, иначе экзамен нечестный",
                    )
                )
    return errors, warnings


def check_kb(path=None, questions="", exam=True, say=print):
    """Проверка базы знаний. → True, если ошибок нет. Экзамен и «мои вопросы» — по рабочему файлу kb/ru.md."""
    from ..personalizer_eval import load_cases, run, summary

    path = Path(path) if path else KB_FILE
    live = path.resolve() == KB_FILE.resolve()
    if live:  # правки файла в этом сеансе Colab вступают в силу сразу
        load_kb.cache_clear()
        default_retriever.cache_clear()
    text = path.read_text(encoding="utf-8")
    fragments = parse_kb(text)
    errors = 0
    say(
        f"База знаний: {path.name}, тем: {len(fragments)}, формулировок: {sum(len(f.examples) for f in fragments)}"
    )

    fmt = format_problems(text)
    for line, fid, msg in fmt:
        say(f"❌ формат, строка {line}, тема {fid}: {msg}")
    errors += len(fmt)
    if not fmt:
        say("✅ формат файла в порядке")

    txt = text_problems(fragments)
    for fid, msg in txt:
        say(f"❌ текст темы {fid}: {msg}")
    errors += len(txt)
    if not txt:
        say(
            f"✅ тексты: нет доз, препаратов, диагнозов и выдуманных цифр (проверено на {len(SAMPLE_CARDS)} ситуациях)"
        )

    cases = load_cases()
    err, warn = example_problems(
        fragments,
        [c["question"] for c in cases if c["split"] == "holdout"],
        [c["question"] for c in cases if c["split"] == "dev"],
    )
    for fid, msg in err:
        say(f"❌ формулировка в теме {fid}: {msg}")
    for fid, msg in warn:
        say(f"⚠️ формулировка в теме {fid}: {msg}")
    errors += len(err)
    if not err:
        say("✅ формулировки не копируют отложенные вопросы экзамена")

    if exam and live and not fmt:
        for split in ("dev", "holdout"):
            part = [c for c in cases if c["split"] == split]
            rows = run(answer, part)
            s = summary(rows, part)
            bad = s["unsafe"]["k"] + s["red_flag_missed"] + s["false_urgent"]
            mark = "❌" if bad else "✅"
            say(
                f"{mark} экзамен {split}: пройдено {s['passed']['k']}/{s['passed']['n']}, опасных {s['unsafe']['k']}, "
                f"пропущено опасных признаков {s['red_flag_missed']}/{s['red_flag_n']}, лишних «срочно» {s['false_urgent']}"
            )
            errors += bad
            if (
                split == "dev"
            ):  # dev — для разработки: показать, что сломалось; holdout — только число
                by_id = {c["id"]: c for c in part}
                for r in rows:
                    if not r["passed"]:
                        say(f"   ✗ {r['id']}: «{by_id[r['id']]['question']}»")
    elif exam and not live:
        say("ℹ️ экзамен не запускался: он работает только с рабочим файлом kb/ru.md")

    asked = [q.strip() for q in (questions or "").split("|") if q.strip()]
    if asked:
        name, photo, lab, profile = DEMO
        say(f"\nВаши вопросы (ситуация: {name}):")
        for q in asked:
            out = answer(q, build_facts(photo, lab, profile))
            say(
                f"• {q}\n  {explain(out)}\n  {out['text'][:220]}{'…' if len(out['text']) > 220 else ''}"
            )

    say(
        "\nИтог: "
        + (
            "✅ ошибок нет"
            if not errors
            else f"❌ ошибок: {errors} — исправьте и запустите ещё раз"
        )
    )
    return errors == 0
