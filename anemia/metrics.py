"""Метрики. Доверительные интервалы: бутстрэп по группам (пациентам) или Уилсон для долей."""
import numpy as np
from sklearn.metrics import roc_auc_score


def iou(pred, true):
    u = (pred | true).sum()
    return float((pred & true).sum() / u) if u else 1.0


def dice(pred, true):
    s = pred.sum() + true.sum()
    return float(2 * (pred & true).sum() / s) if s else 1.0


def wilson(k, n, z=1.96):
    if n == 0:
        return float('nan'), float('nan'), float('nan')
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(p), float(max(0, c - h)), float(min(1, c + h))


def bland_altman(pred, true):
    d = np.asarray(pred) - np.asarray(true)
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    return {'bias': bias, 'loa_lo': bias - 1.96 * sd, 'loa_hi': bias + 1.96 * sd}


def bootstrap_ci(fn, groups, n_boot=1000, seed=0, *arrays):
    """95% ДИ статистики fn(*arrays), ресэмплинг групп с возвращением."""
    rng = np.random.default_rng(seed)
    groups = np.asarray(groups)
    uniq = np.unique(groups)
    idx_of = {g: np.flatnonzero(groups == g) for g in uniq}
    vals = []
    for _ in range(n_boot):
        pick = np.concatenate([idx_of[g] for g in rng.choice(uniq, len(uniq), replace=True)])
        try:
            vals.append(fn(*[np.asarray(a)[pick] for a in arrays]))
        except ValueError:          # в выборку попал один класс
            continue
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def skill(pred, null, true):
    """Доля MAE, которую модель убирает относительно константы: 1 − MAE/MAE₀. 0 — не лучше константы."""
    return 1 - mae(pred, true) / mae(null, true)


def auc(score, y):
    return float(roc_auc_score(y, score))


def threshold_for_sensitivity(score, y, target=0.80):
    """Наибольший порог, при котором доля найденных больных (score ≥ порога) ≥ target. Подбирается на train."""
    pos = np.sort(np.asarray(score)[np.asarray(y).astype(bool)])
    if len(pos) == 0:
        return float('inf')
    k = int(np.floor((1 - target) * len(pos)))      # можно пропустить не больше k больных
    return float(pos[min(k, len(pos) - 1)])
