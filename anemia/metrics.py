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
    ok = [v for v in vals if np.isfinite(v)]     # NaN — метрика не определена на этой выборке (нулевой знаменатель)
    if not ok or len(ok) < 0.95 * len(vals):     # больше 5% таких выборок — ДИ по «удачным» был бы уже честного
        return float('nan'), float('nan')
    vals = ok
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def mae(pred, true):
    return float(np.mean(np.abs(np.asarray(pred) - np.asarray(true))))


def auc(score, y):
    return float(roc_auc_score(y, score))


def threshold_for_sensitivity(score, y, target=0.80):
    """Наибольший порог, при котором доля найденных больных (score ≥ порога) ≥ target. Подбирается на train."""
    pos = np.sort(np.asarray(score)[np.asarray(y).astype(bool)])
    if len(pos) == 0:
        return float('inf')
    k = int(np.floor((1 - target) * len(pos)))      # можно пропустить не больше k больных
    return float(pos[min(k, len(pos) - 1)])


# ------------------------------------------------------------------ бинарный ответ «анемия»
POSITIVE_CLASS = 'анемия (Hb ниже нормы ВОЗ 2011 для пола/возраста/беременности)'


def confusion(flag, y):
    """Матрица ошибок; положительный класс — анемия: y=True — анемия по Hb, flag=True — ответ модели «анемия»."""
    flag, y = np.asarray(flag, bool), np.asarray(y, bool)
    return {'tp': int((flag & y).sum()), 'fp': int((flag & ~y).sum()), 'fn': int((~flag & y).sum()),
            'tn': int((~flag & ~y).sum())}


def _ratio(a, b):
    return a / b if b else float('nan')


def f1_score(flag, y):
    """F1 по классу «анемия»; NaN, если нет ни больных, ни ответов «анемия» (0/0), а не 0."""
    c = confusion(flag, y)
    return float(_ratio(2 * c['tp'], 2 * c['tp'] + c['fp'] + c['fn']))


def balanced_accuracy(flag, y):
    """(чувствительность + специфичность) / 2; NaN, если в выборке один класс."""
    c = confusion(flag, y)
    return float((_ratio(c['tp'], c['tp'] + c['fn']) + _ratio(c['tn'], c['tn'] + c['fp'])) / 2)


def binary_report(flag, y, groups=None, n_boot=1000, seed=0):
    """Метрики бинарного ответа «анемия» (положительный класс — POSITIVE_CLASS) → dict:
        tp, fp, fn, tn — матрица ошибок;
        acc, prec (= PPV), rec (= чувствительность), spec, npv — доли с 95% ДИ Уилсона (_lo, _hi);
        f1, bacc2 (сбалансированная точность двух классов) — с 95% бутстрэп-ДИ по группам (пациентам), если
        groups заданы.
    Метрика с нулевым знаменателем (нет больных, нет здоровых, нет ответов «анемия») — NaN без ДИ: не 0 и не 1.
    Accuracy при дисбалансе классов обманчива (доля здоровых даёт её «бесплатно»): смотреть вместе с bacc2, F1."""
    c = confusion(flag, y)
    tp, fp, fn, tn = c['tp'], c['fp'], c['fn'], c['tn']
    r = dict(c)
    for name, k, n in (('acc', tp + tn, tp + fp + fn + tn), ('prec', tp, tp + fp), ('rec', tp, tp + fn),
                       ('spec', tn, tn + fp), ('npv', tn, tn + fn)):
        r[name], r[name + '_lo'], r[name + '_hi'] = wilson(k, n)
    r['f1'], r['bacc2'] = f1_score(flag, y), balanced_accuracy(flag, y)
    for name, fn_ in (('f1', f1_score), ('bacc2', balanced_accuracy)):
        if groups is not None and n_boot and np.isfinite(r[name]):
            r[name + '_lo'], r[name + '_hi'] = bootstrap_ci(fn_, groups, n_boot, seed, flag, y)
    return r
