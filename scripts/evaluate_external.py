"""Проверка готовой модели ядра (models/hemavision_core.joblib) на внешних наборах без Hb (anemia/external.py).
Ничего не обучает и в results/metrics_history.csv не пишет: это не точность по Hb, а проверка переносимости.

    python scripts/fetch_data.py --external                  # → <data>/external/index.csv
    python scripts/evaluate_external.py --data <data>        # → таблица + <data>/results/external/<run_id>/

Что считается (снимки — через тот же analyze_photo, что в приложении; пол и возраст неизвестны):
    все наборы — сколько снимков принято (status ok) и на каком шаге отказ;
    seg        — сегментатор конъюнктивы модели на всём фото против маски набора: IoU / Dice, доля «нашёл».
Наборы из anemia/external.py EXCLUDED (2026-10-21: guptajanavi — проверка «анемия / нет», welldanyogia — приём
снимков) не оцениваются; их строки в старом index.csv пропускаются. Eyes-defy-копии исключены.
Результаты по снимкам — только id (без имён файлов) — в <data>/results/external/<run_id>/ (вне git).
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anemia import metrics as M  # noqa: E402
from anemia import predict as PR  # noqa: E402
from anemia import runs as RUNS  # noqa: E402
from anemia import segment_baseline as SB  # noqa: E402
from anemia import external as EXT  # noqa: E402
from anemia.io import load_mask, load_rgb, resize_to  # noqa: E402

def _one(root, row, model_path, landmarker):
    B = PR.load_bundle(model_path)
    path = os.path.join(root, row["file"])
    r = PR.analyze_photo(path, bundle=B, landmarker=landmarker)
    out = {
        "status": r["status"],
        "stage": r["stage"],
        "risk": r["risk"],
        "p_anemia": r["p_anemia"],
        "model": r["model"] if r["status"] == "ok" else None,
        "tissues": "+".join(r["tissues"]),
        "why": EXT.safe_text((r["issues"] or [""])[0], [path, row["file"]])[:120],
    }
    # IoU только на фото, которых сегментатор модели не видел при обучении (обученные — scripts/evaluate.py
    # --task seg_ext, вне фолда)
    seen = row["sha1"] in set(B.get("conj_seg_external") or ())
    if isinstance(row.get("mask"), str) and not seen:
        rgb = load_rgb(path)
        gt = resize_to(
            load_mask(os.path.join(root, row["mask"])), rgb.shape[:2], nearest=True
        )
        if gt.any():
            pred = SB.conj_mask(B["conj_seg"], rgb)
            out.update(
                iou=M.iou(pred, gt), dice=M.dice(pred, gt), seg_found=bool(pred.any())
            )
    out["seg_seen"] = seen
    return out


def run(root, idx, model_path, jobs, landmarker):
    res = Parallel(n_jobs=jobs)(
        delayed(_one)(root, r, model_path, landmarker) for r in idx.to_dict("records")
    )
    return pd.concat([idx.reset_index(drop=True), pd.DataFrame(res)], axis=1)


def active_rows(idx, sources=None):
    """Строки индекса для оценки: только наборы из SOURCES (исключённые — пропуск с причиной), без копий Eyes-defy."""
    if sources:
        EXT.check_keys(sources)
    gone = sorted(set(idx.source) & set(EXT.EXCLUDED))
    for key in gone:
        print(f"пропущен {key}: исключён {EXT.EXCLUDED[key]['date']} — {EXT.EXCLUDED[key]['why']}")
    keep = idx.source.isin(sources or list(EXT.SOURCES)) & ~idx.ed_copy.astype(bool)
    return idx[keep].reset_index(drop=True)


def pct(k, n):
    return f"{k}/{n} ({k / n:.0%})" if n else "—"


def report(R):
    lines = [
        "| Набор | Снимков | Групп | Принято | Частые отказы |",
        "|---|---|---|---|---|",
    ]
    for src, g in R.groupby("source"):
        ok = (g.status == "ok").sum()
        top = g[g.status != "ok"].stage.value_counts().head(3)
        lines.append(
            f"| {src} | {len(g)} | {g.group.nunique()} | {pct(ok, len(g))} | "
            + (", ".join(f"{k} {v}" for k, v in top.items()) or "—")
            + " |"
        )
    out = ["## Приём снимков", ""] + lines + [""]
    seg = R[R.use == "seg"]
    s = seg[seg["iou"].notna()] if "iou" in seg else seg.iloc[:0]
    if len(seg):
        out += ["## Сегментатор конъюнктивы на телефонных фото с масками", ""]
        n_seen = int(seg.seg_seen.sum())
        if n_seen:
            out.append(
                f"{n_seen} из {len(seg)} фото были в обучении сегментатора этой модели — здесь IoU по ним не "
                "считается, а их доля «принято» выше — не проверка вне обучения; честный IoU: "
                "`scripts/evaluate.py --task seg_ext`."
            )
        if len(s):
            lo, hi = M.bootstrap_ci(np.mean, s.group.values, 1000, 0, s.iou.values)
            out.append(
                f"Снимков {len(s)}: IoU среднее {s.iou.mean():.2f} [{lo:.2f}–{hi:.2f}], медиана {s.iou.median():.2f}; "
                f"Dice среднее {s.dice.mean():.2f}; конъюнктива найдена на {pct(int(s.seg_found.sum()), len(s))}; "
                f"IoU ≥ 0,5 — {pct(int((s.iou >= 0.5).sum()), len(s))}."
            )
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("ANEMIA_DATA", "data"))
    ap.add_argument(
        "--model",
        default=None,
        help="файл модели (по умолчанию models/hemavision_core.joblib)",
    )
    ap.add_argument(
        "--sources",
        nargs="*",
        help="только эти наборы (ключи anemia/external.py SOURCES)",
    )
    ap.add_argument("--jobs", type=int, default=-1)
    ap.add_argument(
        "--no-face",
        action="store_true",
        help="без MediaPipe (как в облаке, где его нет)",
    )
    a = ap.parse_args()
    warnings.filterwarnings(
        "ignore", message="Only one class is present"
    )  # бутстрэп AUC на крошечных группах
    idx = pd.read_csv(os.path.join(a.data, "external", "index.csv"))
    try:
        idx = active_rows(idx, a.sources)
    except ValueError as e:
        sys.exit(f"evaluate_external: {e}")
    model = a.model or PR.default_model_path()
    B = PR.load_bundle(model)
    print(
        f"Модель: {model} (коммит {B.get('commit')}); снимков: {len(idx)}", flush=True
    )
    R = run(a.data, idx, model, a.jobs, None if a.no_face else "auto")
    run_id, run_dir = RUNS.create_run_dir(
        os.path.join(a.data, "results", "external"), "external"
    )
    R.drop(columns=["file", "mask", "sha1", "dhash"]).to_csv(
        os.path.join(run_dir, "per_image.csv"), index=False
    )
    text = report(R)
    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# Внешние наборы: модель {B.get('commit')}\n\n{text}\n")
    print(text)
    print(f"\nПо снимкам: {run_dir}/per_image.csv")


if __name__ == "__main__":
    main()
