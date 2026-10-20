"""Аудит исходных данных Eyes-defy-anemia и CP-AnemiC (только чтение, исходные файлы не меняются).

    python scripts/audit_data.py --data <dir> [--report results/data_audit.md] [--jobs 4]

Проверяет: битые файлы, точные и почти-дубли снимков, один пациент под разными id, выбросы Hb,
пол/возраст, согласие таблиц с папками, маски врачей против фото (размер, ориентация, положение).
Подробные таблицы по снимкам (с Hb) — в <data>/results/data_audit/ (вне git); в отчёт — только сводка и id.
"""

import argparse
import glob
import hashlib
import os
import re
import struct
import sys
import zlib
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from anemia.data import DHASH_MAX_DIST, THUMB_MAX_MAD, _hamming  # noqa: E402
from anemia.io import load_mask  # noqa: E402

HB_MIN, HB_MAX = 4.0, 20.0  # г/дл: вне — почти наверняка ошибка ввода
NEAR_DHASH = 10  # кандидаты «тот же глаз» для ручной проверки (мягче, чем в манифесте)
NEAR_MAD = 0.03
MASK_DIFF_BAD = (
    20  # медиана |маска − фото| внутри маски, 0–255: > 20 — разметка не с этого кадра
)
cv2.setNumThreads(1)


# ------------------------------------------------------------------ файлы
def sha1_file(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def png_bad_chunks(path):
    """Чанки PNG с неверной CRC (PIL такие файлы не открывает, OpenCV читает с предупреждением)."""
    b = open(path, "rb").read()
    if b[:8] != b"\x89PNG\r\n\x1a\n":
        return ["not-png"]
    bad, i = [], 8
    while i + 12 <= len(b):
        (n,) = struct.unpack(">I", b[i : i + 4])
        t = b[i + 4 : i + 8]
        if i + 12 + n > len(b):
            return bad + ["truncated"]
        if (
            zlib.crc32(b[i + 4 : i + 8 + n])
            != struct.unpack(">I", b[i + 8 + n : i + 12 + n])[0]
        ):
            bad.append(t.decode("latin1"))
        i += 12 + n
        if t == b"IEND":
            return bad
    return bad + ["no-IEND"]


def dhash_arr(gray, size=8):
    a = cv2.resize(gray, (size + 1, size), interpolation=cv2.INTER_AREA).astype(
        np.int16
    )
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int("".join("1" if x else "0" for x in bits), 2)


def thumb_arr(rgb, size=32):
    return (
        cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
        / 255
    )


def file_info(path):
    """Общие проверки файла: читается ли (PIL и OpenCV), размер, режим, CRC PNG."""
    r = {"path": path, "bytes": os.path.getsize(path), "sha1": sha1_file(path)}
    try:
        with Image.open(path) as im:
            im.verify()
        r["pil_ok"] = True
    except Exception as e:  # noqa: BLE001
        r["pil_ok"], r["pil_err"] = False, type(e).__name__
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    r["cv2_ok"] = im is not None
    if im is not None:
        r["h"], r["w"] = im.shape[:2]
        r["channels"] = 1 if im.ndim == 2 else im.shape[2]
    if path.lower().endswith(".png"):
        r["png_bad_crc"] = ",".join(png_bad_chunks(path))
    return r


# ------------------------------------------------------------------ Eyes-defy
ROLE_RE = [
    ("forniceal_palpebral", r"forn\w*ceal_pa\w*ebral"),
    ("forniceal", r"forn\w*ceal"),
    ("palpebral", r"pa\w*ebral"),
]


def mask_role(name):
    n = name.lower()
    for role, pat in ROLE_RE:
        if re.search(pat, n):
            return role
    return None


def _oriented_photo(path, max_side=1200):
    with Image.open(path) as im:
        exif = im.getexif()
        im.draft(
            "RGB", (max_side, max_side)
        )  # быстрое уменьшение JPEG при декодировании
        orient = exif.get(274)
        rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
        return rgb, orient, exif.get(306), exif.get(272)


def ed_folder(d):
    """Одна папка пациента Eyes-defy: все файлы, фото, маски, сверка маски с фото."""
    files = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    out = {"folder": d, "files": [], "masks": []}
    photos = [f for f in files if f.lower().endswith((".jpg", ".jpeg"))]
    for f in files:
        info = file_info(os.path.join(d, f))
        info["role"] = "photo" if f in photos else mask_role(f) or "unknown"
        info["name"] = f
        out["files"].append(info)
    if not photos:
        return out
    p = os.path.join(d, photos[0])
    rgb, orient, dt, model = _oriented_photo(p)
    with Image.open(p) as im:
        raw_w, raw_h = im.size
    ow, oh = (raw_h, raw_w) if orient in (5, 6, 7, 8) else (raw_w, raw_h)
    out.update(
        photo=photos[0],
        n_photos=len(photos),
        exif_orient=orient,
        exif_dt=dt,
        model=model,
        photo_w=ow,
        photo_h=oh,
        dhash=dhash_arr(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)),
        thumb=thumb_arr(rgb),
        lstar=float(cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)[..., 0].mean() / 2.55),
    )
    masks = {}
    for info in out["files"]:
        if info["role"] in ("photo", "unknown") or not info["cv2_ok"]:
            continue
        mp = os.path.join(d, info["name"])
        m = load_mask(mp)
        mrgb = cv2.imread(mp, cv2.IMREAD_COLOR)[..., ::-1]
        mh, mw = m.shape
        ph = cv2.resize(rgb, (mw, mh), interpolation=cv2.INTER_AREA).astype(np.int16)
        diff = lambda a: (
            float(np.median(np.abs(a[m] - mrgb[m].astype(np.int16)).max(-1)))
            if m.any()
            else np.nan
        )
        # та же разметка на неповёрнутом (без EXIF) и перевёрнутом фото — чтобы отличить поворот от чужого кадра
        alt = {"rot180": diff(ph[::-1, ::-1])}
        ys, xs = np.nonzero(m)
        area = m.mean()
        bbox_fill = m.sum() / ((np.ptp(ys) + 1) * (np.ptp(xs) + 1)) if m.any() else np.nan
        ncomp = cv2.connectedComponents(m.astype(np.uint8))[0] - 1
        raw = cv2.imread(mp, cv2.IMREAD_UNCHANGED)
        bg = (
            "alpha"
            if raw.ndim == 3 and raw.shape[2] == 4 and np.median(raw[0, :, 3]) < 128
            else "white"
            if np.median(raw[0, :, :3]) > 200
            else "black"
            if np.median(raw[0, :, :3]) < 30
            else "other"
        )
        rec = {
            "folder": d,
            "name": info["name"],
            "role": info["role"],
            "mask_w": mw,
            "mask_h": mh,
            "aspect_err": abs(mw / mh - ow / oh) / (ow / oh),
            "scale": ow / mw,
            "area": float(area),
            "bbox_fill": float(bbox_fill),
            "n_comp": int(ncomp),
            "bg": bg,
            "diff": diff(ph),
            "diff_rot180": alt["rot180"],
            "cy": float(ys.mean() / mh) if m.any() else np.nan,
            "cx": float(xs.mean() / mw) if m.any() else np.nan,
        }
        out["masks"].append(rec)
        masks.setdefault(info["role"], m)
    if (
        "palpebral" in masks
        and "forniceal_palpebral" in masks
        and masks["palpebral"].shape == masks["forniceal_palpebral"].shape
    ):
        pm, fp = masks["palpebral"], masks["forniceal_palpebral"]
        out["palp_in_fp"] = float((pm & fp).sum() / max(pm.sum(), 1))
    return out


def ed_tables(raw):
    rows = []
    for t in sorted(glob.glob(f"{raw}/**/*.xlsx", recursive=True)):
        country = "India" if "india" in t.lower() else "Italy"
        df = pd.read_excel(t)
        for _, r in df.iterrows():
            hb_raw = r["Hgb"]
            hb = pd.to_numeric(str(hb_raw).replace(",", "."), errors="coerce")
            extra = [
                str(r[c])
                for c in df.columns[4:]
                if pd.notna(r[c]) and str(r[c]).strip()
            ]
            rows.append(
                {
                    "country": country,
                    "pid": str(int(r["Number"])),
                    "hb_raw": hb_raw,
                    "hb_g_dl": hb,
                    "hb_comma": isinstance(hb_raw, str) and "," in hb_raw,
                    "sex": r["Gender"],
                    "age": r["Age"],
                    "note": "; ".join(extra),
                }
            )
    return pd.DataFrame(rows)


def audit_eyes_defy(root, jobs):
    base = os.path.join(root, "eyes_defy_anemia")
    raw = os.path.join(base, "raw")
    folders = sorted(
        os.path.dirname(p) for p in glob.glob(f"{raw}/**/*.jpg", recursive=True)
    )
    folders = sorted(
        set(folders) | {d for d in glob.glob(f"{raw}/*/*/*") if os.path.isdir(d)}
    )
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(ed_folder, folders, chunksize=4))
    files = pd.DataFrame([f for r in res for f in r["files"]])
    masks = pd.DataFrame([m for r in res for m in r["masks"]])
    pat = pd.DataFrame(
        [
            {k: v for k, v in r.items() if k not in ("files", "masks", "thumb")}
            for r in res
        ]
    )
    pat["country"] = pat.folder.map(lambda d: os.path.basename(os.path.dirname(d)))
    pat["pid"] = pat.folder.map(os.path.basename)
    pat["id"] = "ED_" + pat.country + "_" + pat.pid
    thumbs = [r.get("thumb") for r in res]
    tab = ed_tables(raw)
    lab = (
        pd.read_csv(os.path.join(base, "labels.csv"), dtype={"pid": str})
        if os.path.exists(os.path.join(base, "labels.csv"))
        else None
    )
    return {
        "files": files,
        "masks": masks,
        "pat": pat,
        "thumbs": thumbs,
        "tab": tab,
        "labels": lab,
        "raw": raw,
    }


# ------------------------------------------------------------------ CP-AnemiC
def cp_image(path):
    r = file_info(path)
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return r
    rgb = cv2.cvtColor(
        im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR),
        cv2.COLOR_BGR2RGB,
    )
    r["alpha_zero"] = (
        float((im[..., 3] == 0).mean()) if im.ndim == 3 and im.shape[2] == 4 else 0.0
    )
    r["black"] = float((rgb.max(-1) < 10).mean())
    r["white"] = float((rgb.min(-1) > 245).mean())
    r["dhash"] = dhash_arr(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    r["dhash_flip"] = dhash_arr(cv2.cvtColor(rgb[:, ::-1], cv2.COLOR_RGB2GRAY))
    r["thumb"] = thumb_arr(rgb)
    r["mean_rgb"] = rgb.reshape(-1, 3).mean(0).round(1).tolist()
    return r


def audit_cp(root, jobs, xlsx=None):
    base = os.path.join(root, "cp_anemic")
    lab = pd.read_csv(os.path.join(base, "labels.csv"))
    paths = [os.path.join(base, f) for f in lab.file]
    on_disk = sorted(glob.glob(os.path.join(base, "images", "*")))
    with ProcessPoolExecutor(jobs) as ex:
        res = list(ex.map(cp_image, paths, chunksize=8))
    thumbs = [r.pop("thumb", None) for r in res]
    img = pd.DataFrame(res)
    img["file"] = lab.file.values
    img["hb_g_dl"] = lab.hb_g_dl.values
    img["label"] = lab.label.values
    sheets = (
        pd.read_excel(xlsx, sheet_name=None) if xlsx and os.path.exists(xlsx) else None
    )
    return {
        "img": img,
        "thumbs": thumbs,
        "labels": lab,
        "on_disk": on_disk,
        "sheets": sheets,
    }


# ------------------------------------------------------------------ дубли
def pairs(ids, hashes, thumbs, hashes2=None):
    """Пары снимков: точная копия (sha1) — отдельно; здесь dHash ≤ NEAR_DHASH или MAD миниатюр < NEAR_MAD."""
    out = []
    n = len(ids)
    th = np.stack(thumbs)
    for i in range(n):
        mad = np.abs(th[i + 1 :] - th[i]).mean((1, 2, 3))
        for k, j in enumerate(range(i + 1, n)):
            h = _hamming(hashes[i], hashes[j])
            hf = _hamming(hashes2[i], hashes[j]) if hashes2 is not None else 99
            if min(h, hf) <= NEAR_DHASH or mad[k] < NEAR_MAD:
                out.append(
                    {
                        "a": ids[i],
                        "b": ids[j],
                        "dhash": h,
                        "dhash_flip": hf,
                        "mad": float(mad[k]),
                        "is_copy": (h <= DHASH_MAX_DIST and mad[k] < THUMB_MAX_MAD),
                    }
                )
    return pd.DataFrame(out, columns=["a", "b", "dhash", "dhash_flip", "mad", "is_copy"])


def min_mad(thumbs):
    th = np.stack(thumbs)
    best = np.full(len(th), np.inf)
    for i in range(len(th)):
        d = np.abs(th - th[i]).mean((1, 2, 3))
        d[i] = np.inf
        best[i] = d.min()
    return best


# ------------------------------------------------------------------ отчёт
def robust_z(x):
    x = np.asarray(x, float)
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med)) * 1.4826
    return (x - med) / mad if mad > 0 else np.zeros_like(x)


def ids_str(xs, k=12):
    xs = list(xs)
    return (
        ", ".join(map(str, xs[:k])) + (f" … (+{len(xs) - k})" if len(xs) > k else "")
        if xs
        else "—"
    )


def build_report(ed, cp, out_dir):
    problems = []  # (датасет, проблема, сколько, где, последствие, исправление)
    lines = []
    P = lambda *a: problems.append(a)

    if ed is not None:
        files, masks, pat, tab, lab = (
            ed["files"],
            ed["masks"],
            ed["pat"],
            ed["tab"],
            ed["labels"],
        )
        n_jpg = (files.role == "photo").sum()
        n_png = files.name.str.lower().str.endswith(".png").sum()
        lines += [
            "## Eyes-defy-anemia (`data/eyes_defy_anemia`, в запросе data/eda)",
            "",
            f"Папок пациентов: {len(pat)} (Индия {(pat.country == 'India').sum()}, Италия "
            f"{(pat.country == 'Italy').sum()}); файлов: {len(files)} — JPG {n_jpg}, PNG {n_png}. "
            f"Строк в таблицах Hb: {len(tab)}.",
            "",
        ]
        # битые
        bad_cv = files[~files.cv2_ok]
        crc = files[
            files.get("png_bad_crc", pd.Series("", index=files.index))
            .fillna("")
            .str.len()
            > 0
        ]
        pil_only = files[files.cv2_ok & ~files.pil_ok]
        if len(bad_cv):
            P(
                "ED",
                "Файл не читается вообще",
                len(bad_cv),
                ids_str(bad_cv.path.map(os.path.basename)),
                "снимок/маска выпадает",
                "исключить в манифесте, сообщить авторам",
            )
        if len(crc):
            chunks = sorted(set(",".join(crc.png_bad_crc).split(",")))
            P(
                "ED",
                f"PNG-маски с неверной CRC служебного чанка ({', '.join(chunks)}); пиксели целы",
                len(crc),
                f"{crc.folder.nunique() if 'folder' in crc else crc.path.map(os.path.dirname).nunique()} папок, напр. "
                + ids_str(crc.path.map(lambda p: "/".join(p.split(os.sep)[-3:])), 3),
                "PIL/Pillow не открывает (UnidentifiedImageError); OpenCV читает с warning",
                "читать маски только через cv2 (`anemia.io.load_mask` так и делает); для других инструментов — "
                "пересохранённые копии без iCCP в data/derived/, не трогая исходник",
            )
        lines += [
            f"- Читаются OpenCV: {files.cv2_ok.sum()}/{len(files)}; Pillow: {files.pil_ok.sum()}/{len(files)} "
            f"(не открывает {len(pil_only)}, все — PNG с битой CRC: {len(crc)})."
        ]
        # раскладка папок
        photo_cnt = (
            files[files.role == "photo"].groupby(files.path.map(os.path.dirname)).size()
        )
        multi = photo_cnt[photo_cnt > 1]
        no_photo = pat[pat.photo.isna()] if "photo" in pat else pat.iloc[:0]
        roles = files[files.role != "photo"].assign(d=files.path.map(os.path.dirname))
        per = roles.groupby("d").role.apply(list)
        dup_role = [d for d, rs in per.items() if len(rs) != len(set(rs))]
        only_palp = [d for d, rs in per.items() if set(rs) == {"palpebral"}]
        typo = files[
            (files.role.isin(["palpebral", "forniceal_palpebral", "forniceal"]))
            & ~files.name.str.lower().str.contains(
                "palpebral|forniceal$|forniceal\\.", regex=True
            )
        ]
        typo = files[
            files.name.str.lower().str.contains(
                "papebral|palplebral|fornical", regex=True
            )
        ]
        unknown = files[files.role == "unknown"]
        if len(multi):
            P(
                "ED",
                "В папке больше одного фото",
                len(multi),
                ids_str(multi.index.map(lambda d: d[-12:])),
                "берётся первое по алфавиту",
                "проверить вручную",
            )
        if len(no_photo):
            P(
                "ED",
                "Папка без фото",
                len(no_photo),
                ids_str(no_photo.id),
                "пациент выпадает",
                "—",
            )
        if dup_role:
            P(
                "ED",
                "Две версии одной маски (`…(1).png`)",
                len(dup_role),
                ids_str(["ED_" + "_".join(d.split(os.sep)[-2:]) for d in dup_role]),
                "fetch_data берёт `(1)`-копию; она побайтно совпадает с основной (см. sha1) — вреда нет",
                "в fetch_data предпочитать файл без `(1)` (косметика)",
            )
        if len(typo):
            P(
                "ED",
                "Опечатка в имени маски (`papebral`, `palplebral`)",
                len(typo),
                ids_str(typo.path.map(lambda p: "/".join(p.split(os.sep)[-2:]))),
                "fetch_data ищет подстроку `palpebral`: India/7 получает маску forniceal_palpebral (с краем свода), "
                "Italy/95 выпадает из labels.csv целиком",
                "в fetch_data распознавать роль по регулярке `pa\\w*ebral` / `forn\\w*ceal` (как в этом аудите)",
            )
        lines += [
            f"- Состав папок: 4 файла (фото + palpebral + forniceal + forniceal_palpebral) — "
            f"{sum(1 for rs in per if len(rs) == 3)}; только фото + palpebral — {len(only_palp)}; "
            f"с дублем маски — {len(dup_role)}; с нераспознанным файлом — {len(unknown)}."
        ]
        # labels.csv против папок
        if lab is not None:
            lab_ids = set("ED_" + lab.country + "_" + lab.pid.astype(str))
            missing = sorted(set(pat.id) - lab_ids)
            if missing:
                P(
                    "ED",
                    "Папка с фото и маской, но нет в labels.csv",
                    len(missing),
                    ids_str(missing),
                    "пациент не попадает в обучение/оценку",
                    "исправить распознавание масок в fetch_data (см. опечатки)",
                )
            lab_mask = lab.mask_palpebral.fillna("").str.lower()
            wrong_role = lab[lab_mask.str.contains("forniceal")]
            dup_pick = lab[lab_mask.str.contains(r"\(1\)", regex=True)]
            if len(wrong_role):
                P(
                    "ED",
                    "В labels.csv как «маска века» стоит маска forniceal_palpebral",
                    len(wrong_role),
                    ids_str(
                        "ED_" + wrong_role.country + "_" + wrong_role.pid.astype(str)
                    ),
                    "признаки и IoU «конъюнктивы века» считаются по большей области",
                    "см. исправление опечаток",
                )
            if len(dup_pick):
                P(
                    "ED",
                    "В labels.csv выбрана `(1)`-копия маски",
                    len(dup_pick),
                    ids_str("ED_" + dup_pick.country + "_" + dup_pick.pid.astype(str)),
                    "копия побайтно равна основной — на цифры не влияет",
                    "—",
                )
        # таблицы
        tab["id"] = "ED_" + tab.country + "_" + tab.pid
        no_hb = tab[tab.hb_g_dl.isna()]
        comma = tab[tab.hb_comma]
        if len(no_hb):
            P(
                "ED",
                "Нет Hb в таблице",
                len(no_hb),
                ids_str(no_hb.id) + " (" + "; ".join(no_hb.note) + ")",
                "исключается (так и сделано)",
                "—",
            )
        if len(comma):
            P(
                "ED",
                "Hb записан текстом с запятой (`дд,д`) в Italy.xlsx",
                len(comma),
                ids_str(comma.id, 4),
                "наивный `float()` теряет эти строки",
                "парсить с заменой `,`→`.` (fetch_data уже так делает)",
            )
        tab_ids, fold_ids = set(tab.id), set(pat.id)
        only_tab, only_fold = sorted(tab_ids - fold_ids), sorted(fold_ids - tab_ids)
        if only_tab:
            P(
                "ED",
                "Строка в таблице без папки с фото",
                len(only_tab),
                ids_str(only_tab),
                "—",
                "—",
            )
        if only_fold:
            P(
                "ED",
                "Папка без строки в таблице",
                len(only_fold),
                ids_str(only_fold),
                "нет Hb",
                "—",
            )
        hb = tab.hb_g_dl
        out_rng = tab[(hb < HB_MIN) | (hb > HB_MAX)]
        z = pd.Series(np.nan, index=tab.index)
        for s, g in tab.groupby("sex"):
            z[g.index] = robust_z(g.hb_g_dl)
        tab["hb_z"] = z
        hz = tab[tab.hb_z.abs() > 3.5]
        lines += [
            "",
            "**Hb, пол, возраст (таблицы xlsx)**",
            "",
            "| | Индия | Италия |",
            "|---|---|---|",
        ]
        for nm, fn in [
            ("n с Hb", lambda g: str(g.hb_g_dl.notna().sum())),
            (
                "Hb, г/дл: мин–медиана–макс",
                lambda g: (
                    f"{g.hb_g_dl.min():.1f}–{g.hb_g_dl.median():.1f}–"
                    f"{g.hb_g_dl.max():.1f}"
                ),
            ),
            (
                "пол M / F / прочее",
                lambda g: (
                    f"{(g.sex == 'M').sum()} / {(g.sex == 'F').sum()} / "
                    f"{(~g.sex.isin(['M', 'F'])).sum()}"
                ),
            ),
            (
                "возраст: мин–медиана–макс",
                lambda g: f"{g.age.min():.0f}–{g.age.median():.0f}–{g.age.max():.0f}",
            ),
            (
                "возраст пустой / < 18 / > 90",
                lambda g: (
                    f"{g.age.isna().sum()} / {(g.age < 18).sum()} / "
                    f"{(g.age > 90).sum()}"
                ),
            ),
            ("беременность", lambda g: "нет колонки"),
        ]:
            lines.append(
                f"| {nm} | {fn(tab[tab.country == 'India'])} | {fn(tab[tab.country == 'Italy'])} |"
            )
        lines.append("")
        if len(out_rng):
            P(
                "ED",
                f"Hb вне {HB_MIN:.0f}–{HB_MAX:.0f} г/дл",
                len(out_rng),
                ids_str(out_rng.id),
                "ошибка ввода",
                "исключить",
            )
        if len(hz):
            P(
                "ED",
                "Hb — статистический выброс (robust z по модулю > 3,5 внутри пола)",
                len(hz),
                "id — в ed_tables.csv (вне git)",
                "правдоподобно клинически; влияет на MAE",
                "оставить, проверить robust-лоссом; не удалять",
            )
        sex_bad = tab[~tab.sex.isin(["M", "F"])]
        if len(sex_bad):
            P(
                "ED",
                "Пол не M/F",
                len(sex_bad),
                ids_str(sex_bad.id),
                "норма ВОЗ не определяется",
                "уточнить",
            )
        P(
            "ED",
            "Нет поля «беременность»",
            len(tab[tab.sex == "F"]),
            "все женщины",
            "для беременных норма ВОЗ 110, а не 120 г/л → часть «анемий» может быть ложной",
            "хранить pregnant=None (неизвестно), не False; отчитываться с этой оговоркой",
        )
        # один пациент под разными id: одинаковые (страна, пол, возраст, Hb)
        key = (
            tab.dropna(subset=["hb_g_dl"])
            .groupby(["country", "sex", "age", "hb_g_dl"])
            .id.apply(list)
        )
        same = key[key.map(len) > 1]
        lines += [  # только число: id пар вместе с «одинаковым Hb» — производные медданные, в публичный репо нельзя
            f"- Совпадение (страна, пол, возраст, Hb) у разных id: {len(same)} групп "
            f"({sum(map(len, same))} id); id пар — в таблицах <data>/results/data_audit/ (вне git)."
        ]
        # дубли фото
        ok = pat.dhash.notna().values
        ids = pat.id[ok].tolist()
        thumbs = [t for t, k in zip(ed["thumbs"], ok) if k]
        sha = files[files.role == "photo"].groupby("sha1").path.apply(list)
        exact = sha[sha.map(len) > 1]
        sha_m = files[files.role != "photo"].groupby("sha1").path.apply(list)
        exact_m = sha_m[sha_m.map(len) > 1]
        pr = pairs(ids, pat.dhash[ok].astype(int).tolist(), thumbs)
        mm = min_mad(thumbs)
        if len(exact):
            P(
                "ED",
                "Байт-в-байт одинаковые фото",
                len(exact),
                ids_str(exact.map(lambda v: " = ".join(v))),
                "утечка",
                "склеить в group",
            )
        if len(exact_m):
            P(
                "ED",
                "Байт-в-байт одинаковые маски в разных файлах",
                int(exact_m.map(len).sum()),
                ids_str(
                    exact_m.map(
                        lambda v: " = ".join("/".join(p.split(os.sep)[-2:]) for p in v)
                    ),
                    4,
                ),
                "безвредно, если внутри одной папки",
                "брать один файл",
            )
        lines += [
            f"- Дубли фото: sha1 — {len(exact)}; копии по правилу манифеста (dHash ≤ {DHASH_MAX_DIST} и MAD < "
            f"{THUMB_MAX_MAD:.0%}) — {int(pr.is_copy.sum())}; кандидатов «похоже» (dHash ≤ {NEAR_DHASH} или MAD < "
            f"{NEAR_MAD:.0%}) — {len(pr)}. Минимальная MAD до ближайшего другого фото: медиана "
            f"{np.median(mm):.3f}, минимум {mm.min():.3f}."
        ]
        if len(pr):
            pr.to_csv(os.path.join(out_dir, "ed_similar_pairs.csv"), index=False)
            lines.append(
                "  Кандидаты: "
                + ids_str(
                    [
                        f"{a}~{b} (dH {d}, MAD {m:.3f})"
                        for a, b, d, m in zip(pr.a, pr.b, pr.dhash, pr.mad)
                    ],
                    10,
                )
            )
        if pr.is_copy.any():
            c = pr[pr.is_copy]
            P(
                "ED",
                "Почти-копии фото у разных id (возможно один пациент)",
                len(c),
                ids_str(c.a + "~" + c.b),
                "утечка между фолдами",
                "склеить в один group (манифест делает это сам)",
            )
        # время съёмки
        dts = pat.dropna(subset=["exif_dt"]).groupby("exif_dt").id.apply(list)
        dup_dt = dts[dts.map(len) > 1]
        if len(dup_dt):
            P(
                "ED",
                "Одинаковое время съёмки EXIF у разных id",
                len(dup_dt),
                ids_str(dup_dt.map(" = ".join)),
                "скорее всего один снимок/пациент",
                "проверить и склеить в group",
            )
        lines.append(
            f"- Время съёмки EXIF совпадает у разных id: {len(dup_dt)}. Камеры: "
            + ", ".join(f"{k} {v}" for k, v in pat.model.value_counts().items())
            + "."
        )
        # Италия: номер T_n в имени файла против номера папки
        it = pat[(pat.country == "Italy") & pat.photo.fillna("").str.match(r"T_\d+_")]
        if len(it):
            off = it.pid.astype(int) - it.photo.str.extract(r"T_(\d+)_")[0].astype(int)
            lines.append(
                f"- Италия: {len(it)} фото с префиксом `T_n`; номер папки − n = "
                + ", ".join(f"{k} ({v})" for k, v in off.value_counts().items())
                + " — нумерация сдвинута "
                "единообразно, коллизий нет."
                if off.nunique() == 1
                else ""
            )
        # ориентация
        lines.append(
            "- EXIF-ориентация фото: "
            + ", ".join(
                f"{k}: {v}"
                for k, v in pat.exif_orient.fillna(0).astype(int).value_counts().items()
            )
            + " (6 = повернуть на 90°). Маски сохранены уже в повёрнутом виде."
        )
        # маски
        if len(masks):
            masks["id"] = masks.folder.map(
                lambda d: "ED_" + "_".join(d.split(os.sep)[-2:])
            )
            pm = masks[masks.role == "palpebral"]
            lines += [
                "",
                "**Маски врачей против фото** (все маски; «palpebral» — то, что идёт в обучение)",
                "",
                "| Проверка | palpebral | forniceal | forniceal_palpebral |",
                "|---|---|---|---|",
            ]

            def row(nm, fn):
                lines.append(
                    f"| {nm} | "
                    + " | ".join(
                        str(fn(masks[masks.role == r]))
                        if (masks.role == r).any()
                        else "—"
                        for r in ["palpebral", "forniceal", "forniceal_palpebral"]
                    )
                    + " |"
                )

            row("масок", len)
            row(
                "размер (Ш×В)",
                lambda g: ", ".join(
                    f"{w}×{h}: {n}"
                    for (w, h), n in g.groupby(["mask_w", "mask_h"]).size().items()
                ),
            )
            row("пропорции ≠ фото (> 1%)", lambda g: int((g.aspect_err > 0.01).sum()))
            row("масштаб фото/маска, медиана", lambda g: f"{g.scale.median():.2f}")
            row(
                "фон: чёрный / белый / прозрачный",
                lambda g: (
                    f"{(g.bg == 'black').sum()} / {(g.bg == 'white').sum()} / "
                    f"{(g.bg == 'alpha').sum()}"
                ),
            )
            row(
                "|маска − фото| внутри, медиана (0–255)",
                lambda g: f"{g['diff'].median():.1f}",
            )
            row(
                f"не совпадает с фото (> {MASK_DIFF_BAD})",
                lambda g: int((g["diff"] > MASK_DIFF_BAD).sum()),
            )
            row("пустая (< 0,1% кадра)", lambda g: int((g.area < 0.001).sum()))
            row("огромная (> 40% кадра)", lambda g: int((g.area > 0.4).sum()))
            row(
                "прямоугольник (заполнение рамки > 0,95)",
                lambda g: int((g.bbox_fill > 0.95).sum()),
            )
            row("площадь, % кадра: медиана", lambda g: f"{100 * g.area.median():.1f}")
            row("> 3 кусков", lambda g: int((g.n_comp > 3).sum()))
            lines.append("")
            mis = masks[masks["diff"] > MASK_DIFF_BAD]
            if len(mis):
                rot = mis[mis.diff_rot180 < MASK_DIFF_BAD]
                P(
                    "ED",
                    "Маска не совпадает с фото по пикселям (не тот кадр/сдвиг/поворот)",
                    len(mis),
                    ids_str(mis.id + ":" + mis.role),
                    (
                        "из них совпадает после поворота на 180°: " + ids_str(rot.id)
                        if len(rot)
                        else "поворот 180° не объясняет"
                    ),
                    "исключить эти маски из обучения сегментации и из IoU",
                )
            asp = masks[masks.aspect_err > 0.01]
            if len(asp):
                P(
                    "ED",
                    "Пропорции маски ≠ пропорции фото",
                    len(asp),
                    ids_str(asp.id + ":" + asp.role),
                    "resize растянет маску",
                    "исключить/проверить",
                )
            big = masks[masks.area > 0.4]
            if len(big):
                P(
                    "ED",
                    "Маска > 40% кадра (фон не распознан)",
                    len(big),
                    ids_str(big.id + ":" + big.role),
                    "вся картинка считается тканью",
                    "проверить load_mask на этих файлах",
                )
            rect = masks[masks.bbox_fill > 0.95]
            if len(rect):
                P(
                    "ED",
                    "Маска — прямоугольник",
                    len(rect),
                    ids_str(rect.id + ":" + rect.role),
                    "это не контур ткани",
                    "исключить (ED_India_2 уже исключается в error_analysis)",
                )
            empty = masks[masks.area < 0.001]
            if len(empty):
                P(
                    "ED",
                    "Пустая маска",
                    len(empty),
                    ids_str(empty.id + ":" + empty.role),
                    "—",
                    "исключить",
                )
            if "palp_in_fp" in pat:
                sub = pat.dropna(subset=["palp_in_fp"])
                badsub = sub[sub.palp_in_fp < 0.9]
                lines.append(
                    f"- palpebral ⊂ forniceal_palpebral: медиана доли {sub.palp_in_fp.median():.3f} "
                    f"({len(sub)} пар), < 0,9 — {len(badsub)}: {ids_str(badsub.id)}."
                )
                if len(badsub):
                    P(
                        "ED",
                        "Маска века не лежит внутри общей маски (< 90%)",
                        len(badsub),
                        ids_str(badsub.id),
                        "разметки из разных кадров или ошибка",
                        "проверить вручную",
                    )
            masks.to_csv(os.path.join(out_dir, "ed_masks.csv"), index=False)
        pat.drop(columns=["dhash"], errors="ignore").to_csv(
            os.path.join(out_dir, "ed_patients.csv"), index=False
        )
        files.to_csv(os.path.join(out_dir, "ed_files.csv"), index=False)
        tab.to_csv(os.path.join(out_dir, "ed_tables.csv"), index=False)
        lines.append("")

    if cp is not None:
        img, sheets = cp["img"], cp["sheets"]
        lines += [
            "## CP-AnemiC (`data/cp_anemic`, в запросе data/cp)",
            "",
            f"Снимков в labels.csv: {len(img)}; файлов в images/: {len(cp['on_disk'])}. "
            f"Размер вырезов: {img.w.min()}–{img.w.max()} × {img.h.min()}–{img.h.max()} px "
            f"(медиана {img.w.median():.0f}×{img.h.median():.0f}). Каналов: "
            + ", ".join(f"{k}: {v}" for k, v in img.channels.value_counts().items())
            + ".",
            "",
        ]
        bad = img[~img.cv2_ok | ~img.pil_ok]
        if len(bad):
            P(
                "CP",
                "Файл не читается",
                len(bad),
                ids_str(bad.file),
                "выпадает",
                "исключить",
            )
        crc = img[img.png_bad_crc.fillna("").str.len() > 0]
        if len(crc):
            P(
                "CP",
                "PNG с неверной CRC",
                len(crc),
                ids_str(crc.file),
                "—",
                "читать через cv2",
            )
        extra = sorted(
            set(map(os.path.basename, cp["on_disk"]))
            - set(map(os.path.basename, img.file))
        )
        if extra:
            P(
                "CP",
                "Файл в images/ без строки в labels.csv",
                len(extra),
                ids_str(extra),
                "—",
                "—",
            )
        small = img[(img.w < 64) | (img.h < 64)]
        if len(small):
            P(
                "CP",
                "Вырез меньше 64 px по стороне",
                len(small),
                ids_str(small.file),
                "мало пикселей ткани, шумные признаки",
                "помечать низким качеством",
            )
        bg = img[(img.black > 0.2) | (img.white > 0.2) | (img.alpha_zero > 0.2)]
        lines.append(
            f"- Доля фона в вырезе (чёрный < 10, белый > 245, прозрачный): медиана "
            f"{img.black.median():.1%} / {img.white.median():.1%} / {img.alpha_zero.median():.1%}; "
            f"> 20% фона — {len(bg)} вырезов."
        )
        if len(bg):
            P(
                "CP",
                "Вырез на прозрачном фоне (RGB фона = 0), фона > 20% кадра",
                len(bg),
                "все" if len(bg) == len(img) else ids_str(bg.file),
                "загрузка без альфы (`mask_source='full'`) берёт весь кадр; фон отсекает только порог яркости "
                "`tissue_mask` (v > 0,08), полупрозрачный край выреза попадает в признаки",
                "маска = альфа-канал > 0 (эродировать на 1–2 px) вместо «весь кадр»",
            )
        sha = img.groupby("sha1").file.apply(list)
        exact = sha[sha.map(len) > 1]
        pr = pairs(
            img.file.map(os.path.basename).tolist(),
            img.dhash.astype(int).tolist(),
            cp["thumbs"],
            img.dhash_flip.astype(int).tolist(),
        )
        mm = min_mad(cp["thumbs"])
        lines.append(
            f"- Дубли: sha1 — {len(exact)}; копии по правилу манифеста — {int(pr.is_copy.sum())}; "
            f"пар с dHash/зеркальным dHash ≤ {NEAR_DHASH} — {len(pr)} (на гладких вырезах dHash не различает снимки, не показатель); "
            f"из них с MAD < {NEAR_MAD:.0%}: {int((pr.mad < NEAR_MAD).sum())}. "
            f"Минимальная MAD до ближайшего другого выреза: медиана {np.median(mm):.3f}, минимум {mm.min():.3f}."
        )
        if len(pr):
            pr.to_csv(os.path.join(out_dir, "cp_similar_pairs.csv"), index=False)
            close = pr[pr.mad < NEAR_MAD]
            if len(close):
                lines.append(
                    "  Ближайшие: "
                    + ids_str(
                        [
                            f"{a}~{b} (dH {d}, MAD {m:.3f})"
                            for a, b, d, m in zip(
                                close.a, close.b, close.dhash, close.mad
                            )
                        ],
                        10,
                    )
                )
        if len(exact):
            P(
                "CP",
                "Байт-в-байт одинаковые вырезы",
                len(exact),
                ids_str(exact.map(" = ".join)),
                "утечка",
                "склеить",
            )
        if pr.is_copy.any():
            c = pr[pr.is_copy]
            P(
                "CP",
                "Почти-копии вырезов",
                len(c),
                ids_str(c.a + "~" + c.b),
                "утечка",
                "склеить в group",
            )
        hb = img.hb_g_dl
        lab_thr = np.where(hb < 11, "anemic", "non-anemic")
        mism = img[img.label.str.lower().str.replace("_", "-") != lab_thr]
        z = robust_z(hb)
        hz = img[np.abs(z) > 3.5]
        lines += [
            f"- Hb, г/дл: {hb.min():.1f}–{hb.median():.1f}–{hb.max():.1f} (мин–медиана–макс); < 7 (тяжёлая) — "
            f"{(hb < 7).sum()}; вне {HB_MIN:.0f}–{HB_MAX:.0f} — {((hb < HB_MIN) | (hb > HB_MAX)).sum()}; "
            f"|robust z| > 3,5 — {len(hz)}. Метка папки против Hb < 11: расхождений {len(mism)}.",
            f"- Значения Hb с шагом 0,1: уникальных {hb.nunique()} на {len(hb)} снимков.",
        ]
        if len(hz):
            P(
                "CP",
                "Hb — статистический выброс (robust z по модулю > 3,5)",
                len(hz),
                "id — в cp_images.csv (вне git)",
                "клинически возможно (тяжёлая анемия)",
                "оставить; проверить robust-лоссом",
            )
        if len(mism):
            P(
                "CP",
                "Метка папки не совпадает с Hb < 11 г/дл",
                len(mism),
                ids_str(mism.file),
                "шум в метке",
                "брать Hb",
            )
        if sheets is not None:
            s = {k.lower(): v for k, v in sheets.items()}
            grp = (
                pd.concat(
                    [v for k, v in s.items() if "group_id" in v.columns],
                    ignore_index=True,
                )
                if any("group_id" in v.columns for v in s.values())
                else None
            )
            main = next(
                (
                    v
                    for k, v in s.items()
                    if "file_name" in [c.lower() for c in v.columns]
                ),
                None,
            )
            if grp is not None:
                g = grp.group_id.value_counts()
                lines.append(
                    f"- В xlsx есть `group_id` (листы Anaemic/Non-Anaemic): {grp.group_id.nunique()} "
                    f"уникальных на {len(grp)} снимков; повторов {int((g > 1).sum())}."
                )
                if (g > 1).any():
                    P(
                        "CP",
                        "Несколько снимков на один group_id",
                        int((g > 1).sum()),
                        ids_str(g[g > 1].index),
                        "утечка",
                        "использовать group_id как group",
                    )
            if main is not None:
                m = main.rename(columns=str.lower).set_index("file_name").hb_g_dl
                d = img.assign(f=img.file.map(os.path.basename)).set_index("f").hb_g_dl
                diff = (m.reindex(d.index) - d).abs()
                lines.append(
                    f"- labels.csv против xlsx: Hb расходится у {int((diff > 1e-6).sum())}, "
                    f"нет в xlsx — {int(diff.isna().sum())}."
                )
        P(
            "CP",
            "Нет пола и возраста (только «6–59 мес»)",
            len(img),
            "все",
            "норма ВОЗ для < 5 лет (110 г/л) берётся по умолчанию; для 6–59 мес это верно",
            "оставить; отчёт по детям отдельно",
        )
        P(
            "CP",
            "Нет масок врачей и исходных фото (только вырезы)",
            len(img),
            "все",
            "нельзя проверить сегментацию и поиск глаза",
            "использовать только для цвета, не для сегментации",
        )
        P(
            "CP",
            "Один снимок на ребёнка, id = имя файла; связи с пациентом нет",
            len(img),
            "все",
            "«один пациент под разными id» проверяется только по пикселям (см. дубли)",
            "—",
        )
        img.drop(columns=["dhash", "dhash_flip"], errors="ignore").to_csv(
            os.path.join(out_dir, "cp_images.csv"), index=False
        )
        lines.append("")

    tbl = [
        "## Таблица проблем",
        "",
        "| # | Набор | Проблема | Сколько | Где (id) | Последствие | Исправление |",
        "|---|---|---|---|---|---|---|",
    ]
    for i, (ds, what, n, where, eff, fix) in enumerate(problems, 1):
        tbl.append(f"| {i} | {ds} | {what} | {n} | {where} | {eff} | {fix} |")
    return tbl + [""] + lines


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.environ.get("ANEMIA_DATA", "data"))
    ap.add_argument("--report", default="results/data_audit.md")
    ap.add_argument(
        "--cp-xlsx", help="Cp-Anaemic_Dataset.xlsx из архива (листы с group_id)"
    )
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    a = ap.parse_args()
    out_dir = os.path.join(a.data, "results", "data_audit")
    os.makedirs(out_dir, exist_ok=True)
    ed = (
        audit_eyes_defy(a.data, a.jobs)
        if os.path.isdir(os.path.join(a.data, "eyes_defy_anemia", "raw"))
        else None
    )
    xlsx = a.cp_xlsx or next(
        iter(glob.glob(os.path.join(a.data, "cp_anemic", "*.xlsx"))), None
    )
    cp = (
        audit_cp(a.data, a.jobs, xlsx)
        if os.path.exists(os.path.join(a.data, "cp_anemic", "labels.csv"))
        else None
    )
    body = build_report(ed, cp, out_dir)
    head = [
        "# Аудит данных",
        "",
        f"`python scripts/audit_data.py --data <data>` · исходные файлы не изменялись · подробные таблицы — "
        f"`<data>/results/data_audit/` (вне git, содержат Hb).",
        "",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
    with open(a.report, "w", encoding="utf-8") as f:
        f.write("\n".join(head + body) + "\n")
    print(f"отчёт: {a.report}; таблицы: {out_dir}")


if __name__ == "__main__":
    main()
