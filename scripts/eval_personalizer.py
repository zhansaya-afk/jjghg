"""Экзамен AI-персонализатора: прогон по evals/personalizer/cases.json и итоговая таблица.

python scripts/eval_personalizer.py --fn mypkg.module:answer --split dev       # разработка
python scripts/eval_personalizer.py --fn mypkg.module:answer --split holdout   # один раз, в конце
python scripts/eval_personalizer.py --baseline template                         # контроль самого теста
answer(question, facts, history=None) → {'text': str, 'mode': str}; режимы — anemia.personalizer_eval.MODES.
"""

import argparse
import importlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anemia import personalizer_eval as pe

BASELINES = {
    "template": pe.baseline_template,
    "unsafe": pe.baseline_unsafe,
    "confused": pe.baseline_confused,
}


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--fn", help="модуль:функция персонализатора")
    src.add_argument("--baseline", choices=[*BASELINES, "reference"])
    ap.add_argument("--split", choices=[*pe.SPLITS, "all"], default="dev")
    ap.add_argument("--cases", default=str(pe.CASES_PATH))
    ap.add_argument("--out", help="куда записать построчный разбор (JSON)")
    a = ap.parse_args(argv)

    every = pe.load_cases(a.cases)
    errs = pe.validate_cases(
        every
    )  # весь набор: перефразы holdout ссылаются на кейсы dev
    cases = [c for c in every if a.split == "all" or c["split"] == a.split]
    if errs:
        sys.exit("ошибки в наборе кейсов:\n" + "\n".join(errs))
    if a.fn:
        mod, name = a.fn.split(":")
        fn = getattr(importlib.import_module(mod), name)
    else:
        fn = (
            pe.reference_answers(cases)
            if a.baseline == "reference"
            else BASELINES[a.baseline]
        )
    rows = pe.run(fn, cases)
    print(
        pe.format_summary(
            pe.summary(rows, cases),
            f"{a.fn or a.baseline} · {a.split} · {len(cases)} кейсов",
        )
    )
    caught, n_rt, false = pe.red_team()
    print(
        f"самопроверка запретов: поймано {caught}/{n_rt} опасных, ложных срабатываний {len(false)}"
    )
    bad = [r for r in rows if not r["passed"]]
    for r in bad[:15]:
        why = [
            f"{k}={r[k]}"
            for k in ("safety", "forbidden", "invented_numbers", "confused", "missing")
            if r[k]
        ]
        why += [k for k in ("too_long", "too_slow") if r[k]] + (
            [] if r["mode_ok"] else [f"mode={r['mode']}"]
        )
        print(f"  ✗ {r['id']} {r['category']}: {'; '.join(why)}")
    if len(bad) > 15:
        print(f"  … ещё {len(bad) - 15}")
    if a.out:
        Path(a.out).write_text(
            json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
