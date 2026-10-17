"""Анализ ошибок модели v12: сравнение источников маски, матрица ошибок, худшие случаи, разбивка по группам.

Вход — OOF-таблица v12_cv (scripts/evaluate.py): строка = пациент, P — вероятности трёх классов ВОЗ
(норма / лёгкая / умеренная+), thr — порог P(анемия), подобранный внутри обучающих фолдов.
Ничего здесь не подбирает пороги и не исключает снимки: только описывает ошибки на тех же фолдах.
"""
import cv2
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

from . import metrics as M

CLASSES = ('норма', 'лёгкая', 'умеренная+')
SMALL_N = 15


def decide(O, use_thr):
    """Класс v12: argmax (v12.0) или порог P(анемия) (v12.1, выше порога — не ниже «лёгкой»)."""
    P = np.stack(O.P.values)
    k = P.argmax(1)
    if use_thr:
        k = np.where(1 - P[:, 0] >= O.thr.values, np.maximum(k, 1), 0)
    return k


def summary(O, use_thr):
    """n, сбалансированная точность 3 классов, AUC анемии, чувствительность/специфичность (да/нет)."""
    P, y, anem = np.stack(O.P.values), O.y.values, O.anemic.values.astype(bool)
    k = decide(O, use_thr)
    flag = k > 0
    r = {'n': len(O), 'bal_acc': float(balanced_accuracy_score(y, k)) if len(set(y)) > 1 else float('nan')}
    r['auc'] = M.auc(1 - P[:, 0], anem) if 0 < anem.sum() < len(anem) else float('nan')
    r['sens'] = float(flag[anem].mean()) if anem.any() else float('nan')
    r['spec'] = float((~flag[~anem]).mean()) if (~anem).any() else float('nan')
    return r


def paired_delta(A, B, use_thr, n_boot=1000, seed=0):
    """Разница A − B (те же пациенты) по сбалансированной точности и AUC, 95% ДИ бутстрэпом по пациентам."""
    A = A.set_index('group')
    B = B.set_index('group').loc[A.index]
    ka, kb = decide(A, use_thr), decide(B, use_thr)
    pa, pb = 1 - np.stack(A.P.values)[:, 0], 1 - np.stack(B.P.values)[:, 0]
    y, anem = A.y.values, A.anemic.values.astype(bool)
    rng = np.random.default_rng(seed)
    d_ba, d_auc = [], []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) < 3 or anem[i].all() or not anem[i].any():
            continue
        d_ba.append(balanced_accuracy_score(y[i], ka[i]) - balanced_accuracy_score(y[i], kb[i]))
        d_auc.append(M.auc(pa[i], anem[i]) - M.auc(pb[i], anem[i]))
    return {'d_bal_acc': balanced_accuracy_score(y, ka) - balanced_accuracy_score(y, kb),
            'd_bal_acc_ci': tuple(np.percentile(d_ba, [2.5, 97.5])),
            'd_auc': M.auc(pa, anem) - M.auc(pb, anem), 'd_auc_ci': tuple(np.percentile(d_auc, [2.5, 97.5]))}


def confusion(y, k, labels=CLASSES):
    """Строки — правда, столбцы — ответ модели."""
    n = len(labels)
    C = np.zeros((n, n), int)
    for a, b in zip(y, k):
        C[int(a), int(b)] += 1
    return pd.DataFrame(C, index=[f'правда: {c}' for c in labels], columns=[f'ответ: {c}' for c in labels])


def expected_error(O):
    """Ожидаемое расстояние между классами Σ_k P_k·|k − y|: 0 — уверенно верно, 2 — уверенно «норма» при умеренной."""
    P = np.stack(O.P.values)
    return (P * np.abs(np.arange(3)[None] - O.y.values[:, None])).sum(1)


def tertile_labels(x, name, fmt='{:.0f}'):
    """Терцили признака с границами в подписи; NaN → «нет данных»."""
    x = pd.Series(x, dtype=float)
    ok = x.notna()
    out = pd.Series('нет данных', index=x.index, dtype=object)
    if ok.sum() >= 3:
        q = np.nanpercentile(x[ok], [100 / 3, 200 / 3])
        lo, hi = fmt.format(q[0]), fmt.format(q[1])
        out[ok & (x <= q[0])] = f'{name} ≤ {lo}'
        out[ok & (x > q[0]) & (x <= q[1])] = f'{name} {lo}–{hi}'
        out[ok & (x > q[1])] = f'{name} > {hi}'
    return out


def breakdown(Os, meta, by, use_thr=True):
    """Таблица по подгруппам: для каждого источника маски n, сбаланс. точность, AUC, чувств., спец."""
    rows = []
    for name, O in Os.items():
        G = O.merge(meta[['group', by]], on='group')
        for val, sub in G.groupby(by, sort=True):
            r = summary(sub, use_thr)
            rows.append({'признак': by, 'группа': val, 'маска': name, **r,
                         'примечание': 'мало данных' if len(sub) < SMALL_N else ''})
    return pd.DataFrame(rows)


def overlay(crop, mask, color=(0, 255, 0)):
    """Вырезка с контуром маски (uint8 RGB)."""
    img = (np.clip(crop, 0, 1) * 255).astype(np.uint8).copy() if crop.dtype != np.uint8 else crop.copy()
    if mask is not None and mask.any():
        cnt, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, cnt, -1, color, 2)
    return img


def panel(crop, masks, titles):
    """Ряд картинок: исходник + по одной на маску, подписи сверху (латиницей: шрифт OpenCV без кириллицы)."""
    tiles = []
    for m, t in zip([None] + list(masks), titles):
        im = overlay(crop, m)
        bar = np.full((22, im.shape[1], 3), 255, np.uint8)
        cv2.putText(bar, t, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        tiles.append(np.vstack([bar, im]))
    return np.hstack(tiles)


def mask_photo_diff(mask_path, photo):
    """Медиана max|RGB маски − RGB фото| внутри маски врача (0–255). Маска Eyes-defy — это само фото с закрашенным
    фоном, поэтому у исправной разметки ≈ 0; большое значение — испорченная разметка (ED_India_2: белый прямоугольник)."""
    from .io import load_mask, resize_to
    m = load_mask(mask_path)
    im = cv2.imread(mask_path, cv2.IMREAD_COLOR)[..., ::-1].astype(np.int16)
    ph = (resize_to(np.ascontiguousarray(photo, np.float32), m.shape) * 255).round().astype(np.int16)
    return float(np.median(np.abs(ph[m] - im[m]).max(-1))) if m.any() else float('nan')
