"""Внешние наборы фото глаза с Kaggle. Hb в них нет — модель риска на них НЕ обучается (порог ВОЗ без Hb, пола и
возраста не посчитать, а смешивание с Eyes-defy дало бы утечку и завышенные цифры). Зачем они:
    seg    — фото глаза с телефона и маски конъюнктивы: обучение сегментатора века (не модели риска) и его IoU
             вне обучения (scripts/evaluate.py --task seg_ext, фолды по группам).
Исключённые наборы (EXCLUDED) не скачиваются, не индексируются и не оцениваются; причины — там же.
Копии Eyes-defy (COPIES) не скачиваются; файлы, совпавшие с Eyes-defy по sha1 или по миниатюре, помечаются ed_copy
и нигде не используются. Путь к файлу есть только в локальном <data>/external/index.csv (вне git); id, отчёты и
per_image.csv — без имён: id = источник + начало sha1.
"""

import glob
import os

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .data import dhash, sha1_file
from .io import load_mask

IMG_EXT = (".jpg", ".jpeg", ".png")
GROUP_MAD = 0.03  # серия кадров одного глаза: цепочка снимков, соседи в которой ближе 3% (MAD миниатюр 32×32)
ED_COPY_MAD = 0.02  # миниатюра ближе 2% к фото Eyes-defy — пересжатая копия (между разными пациентами ED ≥ 0,065)

SOURCES = {
    "phone_conj_masks": {
        "ref": "mdraselsarker/conjunctiva-segmentation-dataset",
        "use": "seg",
        "licence": "Apache 2.0",
        "what": "83 фото глаза с телефона + маски конъюнктивы (Roboflow)",
    },
}
EXCLUDED = {  # решение команды 2026-10-21 (аудит datasets_audit/2026-10-21_candidates_audit.md)
    "kaggle_conj_binary": {
        "ref": "guptajanavi/palpebral-conjunctiva-to-detect-anaemia",
        "date": "2026-10-21",
        "why": "класс совпадает с источником снимка (анемия — разнородные картинки из интернета, среди них рисунок; "
        "нет — кадры одного видео), Hb и порога нет, лицензия не указана; прежние цифры по нему недействительны",
    },
    "indonesia_phone": {
        "ref": "welldanyogia/anemia-dataset",
        "date": "2026-10-21",
        "why": "в именах файлов ФИО пациентов, лицензия не указана, права на использование не подтверждены; "
        "отключён до подтверждения прав и обезличенной разметки",
    },
}
COPIES = {  # проверено 2026-10-20: каждый файл совпал по sha1 с файлом Eyes-defy (scripts/eyes_defy.sha1)
    "kpriyadharcini/anaemia": "122 из 122 фото — побайтные копии Eyes-defy",
    "abood92/eye-segmentation": "434 из 434 файлов (217 фото + 217 масок) — побайтные копии Eyes-defy",
    "abhit007pandey/eye-anemia": "архив того же размера, что Eyes-defy (666 269 625 байт)",
}


def check_keys(keys):
    """Ключи наборов для скачивания и индекса; исключённые и неизвестные — ValueError с причиной."""
    for key in keys:
        if key in EXCLUDED:
            e = EXCLUDED[key]
            raise ValueError(
                f"набор {key} ({e['ref']}) исключён {e['date']}: {e['why']}"
            )
        if key not in SOURCES:
            raise ValueError(f"неизвестный набор {key}; доступны: {', '.join(SOURCES)}")
    return list(keys)


def kaggle_url(ref):
    return f"https://www.kaggle.com/api/v1/datasets/download/{ref}"


def thumb(path, size=32):
    """Миниатюра 32×32 RGB [0, 1] с учётом EXIF; JPEG декодируется сразу уменьшенным (draft) — быстро и для 12 Мп."""
    with Image.open(path) as im:
        im.draft("RGB", (size * 8, size * 8))
        im = ImageOps.exif_transpose(im).convert("RGB")
        return np.asarray(im.resize((size, size), Image.BILINEAR), np.float32) / 255


def _images(raw):
    return sorted(
        f
        for f in glob.glob(os.path.join(raw, "**", "*"), recursive=True)
        if f.lower().endswith(IMG_EXT) and os.path.isfile(f)
    )


def list_source(key, raw):
    """Файлы набора → [{file, label, site, mask}] (пути абсолютные). label: 1 анемия, 0 нет, None нет метки."""
    check_keys([key])
    rows = []
    if key == "phone_conj_masks":
        for f in _images(raw):
            if f"{os.sep}images{os.sep}" not in f:
                continue
            stem = os.path.splitext(
                f.replace(f"{os.sep}images{os.sep}", f"{os.sep}masks{os.sep}")
            )[0]
            masks = [
                m
                for m in glob.glob(glob.escape(stem) + ".*")
                if m.lower().endswith(IMG_EXT)
            ]
            rows.append(
                {
                    "file": f,
                    "label": None,
                    "site": f.split(os.sep)[-3],
                    "mask": masks[0] if masks else None,
                }
            )
    else:
        raise KeyError(key)
    return rows


def near_groups(thumbs, ids, max_mad=GROUP_MAD):
    """Группы почти одинаковых кадров: single linkage по MAD миниатюр — снимки в одной группе, если их связывает
    цепочка соседей ближе max_mad. Осторожное направление: лишняя склейка только уменьшает число групп, а
    недосклейка дала бы кадры одного глаза и в train, и в test (и лишние «голоса» в binary). Имя — первый id группы."""
    n = len(ids)
    if n < 2:
        return list(ids)
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    th = np.stack(thumbs).reshape(n, -1)
    D = np.abs(th[:, None, :] - th[None, :, :]).mean(-1)
    lab = fcluster(linkage(squareform(D, checks=False), "single"), max_mad, "distance")
    first = {}
    for i, c in enumerate(lab):
        first.setdefault(c, ids[i])
    return [first[c] for c in lab]


def build_index(root, keys=None, eyes_defy_sha1=(), eyes_defy_thumbs=None):
    """<root>/external/<key>/raw → таблица: source, use, id, file (относительно root, только локально), label,
    site, mask, mask_frac (доля площади маски), sha1, dhash, group, ed_copy. ed_copy: файл побайтно есть в Eyes-defy (sha1) или его миниатюра ближе
    ED_COPY_MAD к фото Eyes-defy (eyes_defy_thumbs, если переданы) — такие снимки нигде не используются."""
    ed = set(eyes_defy_sha1)
    ed_th = (
        None
        if eyes_defy_thumbs is None or not len(eyes_defy_thumbs)
        else np.stack(eyes_defy_thumbs)
    )
    parts = []
    for key in check_keys(keys or SOURCES):
        raw = os.path.join(root, "external", key, "raw")
        if not os.path.isdir(raw):
            continue
        rows = list_source(key, raw)
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["sha1"] = df.file.map(sha1_file)
        df = df.drop_duplicates("sha1").reset_index(
            drop=True
        )  # побайтные дубли внутри набора — один снимок
        df["id"] = key + "_" + df.sha1.str[:10]
        th = [thumb(f) for f in df.file]
        df["group"] = near_groups(th, df.id.tolist())
        df["dhash"] = [f"{dhash(f):016x}" for f in df.file]
        near_ed = (
            np.zeros(len(df), bool)
            if ed_th is None
            else np.array(
                [np.abs(ed_th - t).mean((1, 2, 3)).min() < ED_COPY_MAD for t in th]
            )
        )
        df["ed_copy"] = df.sha1.isin(ed).values | near_ed
        df["mask_frac"] = [
            float(load_mask(m).mean()) if isinstance(m, str) else np.nan
            for m in df["mask"]
        ]
        df["source"], df["use"] = key, SOURCES[key]["use"]
        for c in ("file", "mask"):
            df[c] = df[c].map(
                lambda p: os.path.relpath(p, root) if isinstance(p, str) else None
            )
        parts.append(df)
    cols = [
        "source",
        "use",
        "id",
        "file",
        "label",
        "site",
        "mask",
        "mask_frac",
        "sha1",
        "dhash",
        "group",
        "ed_copy",
    ]
    return (
        pd.concat(parts, ignore_index=True)[cols]
        if parts
        else pd.DataFrame(columns=cols)
    )


def summary(index):
    """Сводка по наборам: снимков, групп (разных серий; людей не больше), групп по классам, с маской, копий ED."""
    g = index.groupby("source")
    by_cls = lambda v: g.apply(
        lambda d: d.loc[d.label == v, "group"].nunique(), include_groups=False
    )  # noqa: E731
    return pd.DataFrame(
        {
            "снимков": g.size(),
            "групп": g.group.nunique(),
            "групп_анемия": by_cls(1),
            "групп_нет": by_cls(0),
            "с_маской": g["mask"].apply(lambda s: int(s.notna().sum())),
            "копий_ED": g.ed_copy.sum().astype(int),
        }
    )


def seg_index(root):
    """Строки индекса для обучения сегментатора: use == 'seg', маска есть и не пустая, не копия Eyes-defy."""
    path = os.path.join(root, "external", "index.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    idx = pd.read_csv(path)
    return idx[
        (idx["use"] == "seg")
        & idx["mask"].notna()
        & (idx.get("mask_frac", pd.Series(np.nan, idx.index)) > 0)
        & ~idx.ed_copy.astype(bool)
    ].reset_index(drop=True)


def safe_text(text, paths):
    """Сообщение без путей к файлам (в именах файлов внешних наборов бывают ФИО)."""
    text = str(text or "")
    for p in paths:
        if p:
            text = text.replace(str(p), "<фото>").replace(
                os.path.basename(str(p)), "<фото>"
            )
    return text
