"""Модель демо-блокнота v12 (notebooks/anemia_screening_v12.ipynb), перенесённая без изменений логики,
чтобы scripts/evaluate.py оценивал её на фиксированных фолдах манифеста, как все остальные методы.

Что делает v12: ткани ищет anemia.regions.detect_regions (режим 'both'), признаки цвета считает своей
roi_features (OpenCV Lab/HSV — именно с ней посчитан cp_anemic_features.npz), классификатор — HGB на три
класса ВОЗ (норма / лёгкая / умеренная+) с уравниванием классов повтором и байесовской поправкой на частоту.
Ответ блокнота — argmax; v12.1 (PR #6) вместо этого сравнивает P(анемия) с порогом на чувствительность 0,80.
"""
import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .regions import detect_regions

EPS = 1e-4
N_RELIGHT = 3          # копий каждого фото при другом свете (как в v12)
DETECT_SIDE = 1024     # длинная сторона фото перед поиском тканей (как в v12, _prep_one)
CROP_SIDE = 256        # вырезы CP-AnemiC: как признаки в cp_anemic_features.npz (v11 WORK_SIZE)
MODES = ('conj', 'sclera', 'both')


# ------------------------------------------------------------------ пороги ВОЗ (г/дл), как в v12
def who_thresholds(sex, age, pregnant=False):
    """(граница нормы, граница умеренной), г/дл."""
    if age is not None and age < 5:
        return 11.0, 10.0
    if pregnant:
        return 11.0, 10.0
    if age is not None and age < 12:
        return 11.5, 11.0
    if age is not None and age < 15:
        return 12.0, 11.0
    return (13.0, 11.0) if sex == 'M' else (12.0, 11.0)


def hb_class(hb_g_dl, thr_norm, thr_mod):
    return 0 if hb_g_dl >= thr_norm else 1 if hb_g_dl >= thr_mod else 2


# ------------------------------------------------------------------ цвет и свет
def srgb_to_lin(x):
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055)


def relight(rgb, rng, strength=1.0):
    """Обучающий «другой свет» v12: баланс белого, экспозиция, гамма."""
    lin = srgb_to_lin(np.clip(rgb, 0, 1))
    t = rng.normal(0, 0.18 * strength)
    tint = rng.normal(0, 0.06 * strength)
    gains = np.exp(np.array([t, tint, -t], np.float32))
    expo = np.exp(rng.normal(0, 0.30 * strength))
    out = lin_to_srgb(lin * gains * expo)
    return (np.clip(out, 0, 1) ** np.exp(rng.normal(0, 0.10 * strength))).astype(np.float32)


def white_balance_by(rgb, ref_mask):
    lin = srgb_to_lin(np.clip(rgb, 0, 1))
    ref = np.median(lin[ref_mask], axis=0) + 1e-4
    lin = lin * (ref.mean() / ref)
    return lin_to_srgb(lin).astype(np.float32)


def tissue_mask(rgb, mask):
    v = rgb.max(-1)
    s = (v - rgb.min(-1)) / (v + EPS)
    m = mask & (v > 0.08) & ~((v > 0.92) & (s < 0.25))
    if m.sum() < 30:
        return mask & (v > 0.03)
    lo, hi = np.percentile(v[m], [3, 99])
    return m & (v >= lo) & (v <= hi)


def roi_features(rgb, mask, prefix):
    m = tissue_mask(rgb, mask)
    if m.sum() < 30:
        return None
    px = np.clip(rgb[m].astype(np.float32), 1 / 255, 1)
    R, G, B = px[:, 0], px[:, 1], px[:, 2]
    lab = cv2.cvtColor(px[None], cv2.COLOR_RGB2Lab)[0]
    L, A, Bb = lab[:, 0], lab[:, 1], lab[:, 2]
    hsv = cv2.cvtColor(px[None], cv2.COLOR_RGB2HSV)[0]
    hue = np.where(hsv[:, 0] > 180, hsv[:, 0] - 360, hsv[:, 0])
    S = R + G + B
    ch = {'EI': np.log10(R) - np.log10(G),
          'lRB': np.log(R / B), 'lGB': np.log(G / B),
          'r': R / S, 'g': G / S, 'a': A, 'b': Bb, 'L': L, 'hue': hue, 'sat': hsv[:, 1],
          'C': np.hypot(A, Bb), 'hab': np.degrees(np.arctan2(Bb, A))}
    f = {}
    for k, v in ch.items():
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        f[k + '_m'], f[k + '_p10'], f[k + '_p50'], f[k + '_p90'], f[k + '_iqr'] = \
            float(v.mean()), float(p10), float(p50), float(p90), float(p90 - p10)
    ei = ch['EI']
    f['HHR'] = float(((hue > -10) & (hue < 10)).mean())
    f['vessel'] = float(np.percentile(ei, 95) - np.median(ei))
    f['a_over_L'] = float(np.median(A) / (np.median(L) + EPS))
    return {prefix + k: v for k, v in f.items()}


def eye_features(rgb, conj, sclera):
    """c_ — конъюнктива, s_ — склера, w_ — конъюнктива после баланса белого по склере."""
    out = {}
    fc = roi_features(rgb, conj, 'c_') if conj is not None and conj.sum() >= 30 else None
    fs = roi_features(rgb, sclera, 's_') if sclera is not None and sclera.sum() >= 30 else None
    if fc:
        out.update(fc)
    if fs:
        out.update(fs)
    if fc and fs:
        fw = roi_features(white_balance_by(rgb, tissue_mask(rgb, sclera)), conj, 'w_')
        if fw:
            out.update(fw)
    return out


def detect(rgb):
    """→ (вырезка, маска конъюнктивы, маска склеры) поиском v12 в режиме «конъюнктива и склера»."""
    det = detect_regions(rgb, 'both')
    return det['crop'], det['conj'], det['sclera']


def feature_sets(columns, has_sex):
    """Наборы признаков трёх моделей взрослых, как в train_all() блокнота."""
    c = [k for k in columns if k.startswith('c_')]
    s = [k for k in columns if k.startswith('s_')]
    w = [k for k in columns if k.startswith('w_')]
    m = ['male'] if has_sex else []
    return {'conj': c + m, 'sclera': s + m, 'both': c + s + w + m}


# ------------------------------------------------------------------ классификатор
def _clf():
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                          l2_regularization=1.0, min_samples_leaf=15, random_state=0)


def fit(X, y, seed=0):
    """Классы уравниваются повторением редких примеров (sample_weight в sklearn ≥ 1.9 в 30 раз медленнее)."""
    rng = np.random.default_rng(seed)
    cls, cnt = np.unique(y, return_counts=True)
    idx = np.concatenate([np.concatenate([np.flatnonzero(y == c),
                                          rng.choice(np.flatnonzero(y == c), cnt.max() - n, replace=True)])
                          for c, n in zip(cls, cnt)])
    return _clf().fit(X[idx], y[idx])


def proba3(m, X):
    P = np.zeros((len(X), 3))
    P[:, m.classes_.astype(int)] = m.predict_proba(X)
    return P


def class_prior(y):
    return (np.bincount(np.asarray(y, int), minlength=3) + 1) / (len(y) + 3)


def prior_correct(P, prior):
    Q = np.asarray(P) * np.asarray(prior)[None]
    return Q / Q.sum(-1, keepdims=True)


def predict(m, X, prior):
    """Вероятности трёх классов с поправкой на частоту классов обучения — как при анализе нового фото."""
    return prior_correct(proba3(m, X), prior)
