"""Признаки цвета по строкам манифеста: исходный снимок, обучающие копии при случайном свете, стресс-условия."""
import os

import numpy as np

from .augment import STRESS_CONDITIONS, apply_illuminant, random_relight
from .color import roi_features
from .io import load_mask, load_rgb, resize_to
from .labs import who_limits

WORK_SIZE = 256        # длинная сторона снимка для признаков (как в v11)
# Обучающих копий при случайном свете на снимок. В v11 было 3; на CP-AnemiC 12 копий дают AUC +0,01–0,02
# и MAE −0,3…−0,7 г/л при трёх разных seed (docs/quality_roadmap.md). Первые 3 копии совпадают с v11:
# генератор тот же, поэтому сравнение «3 против 12» парное. Базу v11 воспроизводит evaluate.py --relight 3.
N_RELIGHT = 12


def load_item(root, row, mask_source):
    """→ (rgb, маска конъюнктивы). mask_source: 'full' — весь кадр (CP-AnemiC уже вырезан), 'doctor' — маска врача."""
    rgb = load_rgb(os.path.join(root, row['image']), WORK_SIZE)
    if mask_source == 'full':
        return rgb, np.ones(rgb.shape[:2], bool)
    if mask_source == 'doctor':
        return rgb, resize_to(load_mask(os.path.join(root, row['mask_conj'])), rgb.shape[:2], nearest=True)
    raise ValueError(mask_source)


def row_meta(row):
    age = row['age'] if row['age'] == row['age'] else None
    sex = row['sex'] if isinstance(row['sex'], str) else None
    return {'group': row['group'], 'fold': int(row['fold']), 'hb_g_l': float(row['hb_g_l']),
            'anemic': bool(row['anemic']), 'normal_thr': who_limits(sex, age, bool(row['pregnant'])).normal,
            'male': 1.0 if sex == 'M' else 0.0 if sex == 'F' else 0.5}


def feature_rows(rgb, mask, meta, seed, n_relight=N_RELIGHT, stress=True):
    """→ (обучающие строки [aug 0..n], строки стресс-теста [cond]). Пустые, если на маске нет ткани."""
    f0 = roi_features(rgb, mask, 'c_')
    if f0 is None:
        return [], []
    rows = [{**meta, 'aug': 0, **f0}]
    rng = np.random.default_rng(seed)
    for a in range(1, n_relight + 1):
        f = roi_features(random_relight(rgb, rng), mask, 'c_')
        if f is not None:
            rows.append({**meta, 'aug': a, **f})
    srows = []
    if stress:
        for c, (T, e) in enumerate(STRESS_CONDITIONS):
            f = roi_features(apply_illuminant(rgb, T, e), mask, 'c_')
            if f is not None:
                srows.append({**meta, 'cond': f'{T}K×{e}', **f})
    return rows, srows
