"""Единый манифест данных: одна строка — один снимок.

Колонки: dataset, patient_id, group, image, mask_conj, hb_g_l, sex, age, pregnant, population,
anemic, grade, sha1, dhash, fold. Пути — относительно корня данных (по умолчанию data/).
group объединяет снимки одного пациента и почти-дубли (dHash) — по нему делятся фолды,
поэтому один человек или копия одного фото никогда не попадает одновременно в train и test.
"""
import glob
import hashlib
import os

import numpy as np
import pandas as pd
from PIL import Image

from .labs import is_anemic, is_pregnant, norm_pregnant, who_grade

COLUMNS = ['dataset', 'patient_id', 'group', 'image', 'mask_conj', 'hb_g_l', 'sex', 'age', 'pregnant',
           'population', 'anemic', 'grade', 'sha1', 'dhash', 'fold']
N_FOLDS = 5
DHASH_MAX_DIST = 3     # кандидат в копию: dHash отличается не более чем на 3 бита из 64
THUMB_MAX_MAD = 0.01   # и подтверждение: средняя разница пикселей миниатюр 32×32 < 1%
# Только dHash на гладких вырезах конъюнктивы склеивает разные снимки: проверено 2026-10-14 на вырезах CP-AnemiC
# (датасет с 2026-10-19 исключён), 0 из 243 пар с dHash ≤ 3 не были копиями (MAD ≥ 0,02), поэтому копией
# считается только пара, прошедшая оба теста.


# Маски врачей, по которым сегментатор не учится (фото и Hb пациента остаются): аудит данных 2026-10-19,
# scripts/audit_data.py — у ED_India_2 внутри маски кусок другого кадра (|маска − фото| 73 из 255, у остальных ≈ 1).
BAD_DOCTOR_MASKS = {'ED_India_2': 'маска не с этого кадра'}


# ------------------------------------------------------------------ хеши
def sha1_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def dhash(path, size=8):
    """Разностный хеш 64 бит: устойчив к пересжатию и масштабу, различает разные снимки."""
    im = Image.open(path).convert('L').resize((size + 1, size), Image.BILINEAR)
    a = np.asarray(im, np.int16)
    bits = (a[:, 1:] > a[:, :-1]).flatten()
    return int(''.join('1' if b else '0' for b in bits), 2)


def _hamming(a, b):
    return bin(a ^ b).count('1')


def thumb(path, size=32):
    return np.asarray(Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR), np.float32) / 255


# ------------------------------------------------------------------ нормализация полей
def norm_sex(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    s = str(v).strip().upper()
    if s[:1] in ('M', 'U'):          # M, Male, Uomo
        return 'M'
    if s[:1] in ('F', 'W', 'D'):     # F, Female, Woman, Donna
        return 'F'
    return None


def _num(v):
    x = pd.to_numeric(str(v).replace(',', '.'), errors='coerce') if v is not None else np.nan
    return float(x) if pd.notna(x) else np.nan


# ------------------------------------------------------------------ источники
def load_eyes_defy(root):
    """data/eyes_defy_anemia/: raw/ + labels.csv из scripts/fetch_data.py. Взрослые, Hb в г/дл."""
    d = os.path.join(root, 'eyes_defy_anemia')
    lab = pd.read_csv(os.path.join(d, 'labels.csv'), dtype={'pid': str})
    rel = lambda p: os.path.join('eyes_defy_anemia', 'raw', p) if isinstance(p, str) and p else None
    return pd.DataFrame({
        'dataset': 'eyes_defy',
        'patient_id': 'ED_' + lab.country.astype(str) + '_' + lab.pid.astype(str),
        'image': lab.image.map(rel),
        'mask_conj': lab.mask_palpebral.map(rel),
        'hb_g_l': lab.hb_g_dl.map(_num) * 10,
        'sex': lab['sex'].map(norm_sex) if 'sex' in lab else None,
        'age': lab['age'].map(_num) if 'age' in lab else np.nan,
        'pregnant': None,                   # в Eyes-defy поля беременности нет — статус неизвестен, не False
        'population': 'adult'})


def load_phone(root):
    """data/phone/: снимки + hb.csv (file, patient_id, hb_g_l, sex, age, pregnant[, mask_conj])."""
    d = os.path.join(root, 'phone')
    lab = pd.read_csv(os.path.join(d, 'hb.csv'), dtype={'patient_id': str})
    get = lambda c, default=None: lab[c] if c in lab else default
    return pd.DataFrame({
        'dataset': 'phone',
        'patient_id': 'PH_' + lab.patient_id.astype(str),
        'image': lab.file.map(lambda f: os.path.join('phone', f)),
        'mask_conj': get('mask_conj', pd.Series([None] * len(lab))).map(
            lambda f: os.path.join('phone', f) if isinstance(f, str) and f else None),
        'hb_g_l': lab.hb_g_l.astype(float),
        'sex': get('sex', pd.Series([None] * len(lab))).map(norm_sex),
        'age': get('age', pd.Series([np.nan] * len(lab))).map(_num),
        'pregnant': get('pregnant', pd.Series([None] * len(lab))).map(norm_pregnant).astype(object),
        'population': np.where(get('age', pd.Series([99] * len(lab))).map(_num) < 15, 'child', 'adult')})


LOADERS = {'eyes_defy': ('eyes_defy_anemia/labels.csv', load_eyes_defy),
           'phone': ('phone/hb.csv', load_phone)}       # CP-AnemiC исключён 2026-10-19


# ------------------------------------------------------------------ группы и фолды
def _groups(df, thumbs=None):
    """Union-find: один пациент → одна группа; почти-дубли (dHash) склеиваются в ту же группу."""
    parent = list(range(len(df)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        parent[find(i)] = find(j)

    first = {}
    for i, p in enumerate(df.patient_id):
        if p in first:
            union(i, first[p])
        first.setdefault(p, i)
    hs = df.dhash.tolist()
    for i in range(len(hs)):
        for j in range(i + 1, len(hs)):
            if _hamming(hs[i], hs[j]) <= DHASH_MAX_DIST and (
                    thumbs is None or np.abs(thumbs[i] - thumbs[j]).mean() < THUMB_MAX_MAD):
                union(i, j)
    roots = [find(i) for i in range(len(df))]
    names = {}
    for i, r in enumerate(roots):
        names.setdefault(r, df.patient_id.iloc[i])
    return [names[r] for r in roots]


def assign_folds(df, n_folds=N_FOLDS, seed=0):
    """Фолды отдельно по каждому датасету: StratifiedGroupKFold по анемии (GroupKFold, если класс один)."""
    from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
    fold = pd.Series(-1, index=df.index)
    for _, part in df.groupby('dataset'):
        y = part.anemic.astype(int).values
        k = min(n_folds, part.group.nunique())
        stratify = len(set(y)) > 1
        if stratify:
            k = min(k, np.bincount(y).min())
        if k < 2:
            fold[part.index] = 0
            continue
        cv = (StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed) if stratify
              else GroupKFold(n_splits=k))
        for f, (_, te) in enumerate(cv.split(part, y, part.group)):
            fold[part.index[te]] = f
    return fold.astype(int)


def build_manifest(root, datasets=None, n_folds=N_FOLDS, seed=0):
    parts = []
    for name, (marker, loader) in LOADERS.items():
        if datasets and name not in datasets:
            continue
        if os.path.exists(os.path.join(root, marker)):
            parts.append(loader(root))
    if not parts:
        raise FileNotFoundError(f'В {root} нет ни одного датасета (ожидаются: '
                                + ', '.join(m for m, _ in LOADERS.values()) + ')')
    df = pd.concat(parts, ignore_index=True)
    df['pregnant'] = df.pregnant.map(norm_pregnant).astype(object)     # None = неизвестно
    df = df[df.image.notna() & df.hb_g_l.notna()].reset_index(drop=True)
    exists = df.image.map(lambda p: os.path.exists(os.path.join(root, p)))
    if (~exists).any():
        print(f'⚠️ нет файлов: {(~exists).sum()} (пропущены), напр. {df.image[~exists].iloc[0]}')
        df = df[exists].reset_index(drop=True)
    df['sha1'] = df.image.map(lambda p: sha1_file(os.path.join(root, p)))
    df['dhash'] = df.image.map(lambda p: dhash(os.path.join(root, p)))
    df['anemic'] = [is_anemic(h, s, a if pd.notna(a) else None, is_pregnant(p))
                    for h, s, a, p in zip(df.hb_g_l, df.sex, df.age, df.pregnant)]
    df['grade'] = [who_grade(h, s, a if pd.notna(a) else None, is_pregnant(p))
                   for h, s, a, p in zip(df.hb_g_l, df.sex, df.age, df.pregnant)]
    thumbs = [thumb(os.path.join(root, p)) for p in df.image]
    df['group'] = _groups(df, thumbs)
    df['fold'] = assign_folds(df, n_folds, seed)
    df['dhash'] = df.dhash.map(lambda h: f'{h:016x}')
    return df[COLUMNS]


def check_no_leak(df):
    """Ни одна группа и ни один sha1 не встречается в двух фолдах одного датасета."""
    for col in ('group', 'sha1'):
        n = df.groupby(['dataset', col]).fold.nunique()
        if (n > 1).any():
            raise AssertionError(f'утечка: {col} в нескольких фолдах: {list(n[n > 1].index[:5])}')
    cross = df.groupby('group').dataset.nunique()
    return list(cross[cross > 1].index)     # почти-дубли между датасетами — предупреждение, не ошибка
