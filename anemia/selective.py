"""Зона «не уверен»: два порога P(анемия), подобранные только на обучающих фолдах.

P ≥ t_hi → «риск анемии высокий», P ≤ t_lo → «признаков анемии нет», между ними → «не уверен, сдайте анализ крови».
Пороги — по вложенным OOF-вероятностям обучающих фолдов: t_hi — наименьший, при котором среди P ≥ t_hi доля
больных ≥ target; t_lo — наибольший, при котором среди P ≤ t_lo доля здоровых ≥ target.
"""
import numpy as np


def zone_thresholds(p, y, target=0.90, min_n=5):
    p, y = np.asarray(p, float), np.asarray(y).astype(bool)
    order = np.sort(np.unique(p))
    t_hi, t_lo = np.inf, -np.inf
    for t in order:                                   # от низких к высоким: первый, где «да» достаточно чисто
        sel = p >= t
        if sel.sum() >= min_n and y[sel].mean() >= target:
            t_hi = t
            break
    for t in order[::-1]:                             # от высоких к низким: первый, где «нет» достаточно чисто
        sel = p <= t
        if sel.sum() >= min_n and (~y[sel]).mean() >= target:
            t_lo = t
            break
    if t_lo >= t_hi:                                  # зоны пересеклись — разделить посередине
        t_lo = t_hi = (t_lo + t_hi) / 2
    return float(t_lo), float(t_hi)


def zone_metrics(p, y, t_lo, t_hi):
    """→ доля уверенных ответов, точность на них, чувств./спец. на них, доля больных среди «не уверен»."""
    p, y = np.asarray(p, float), np.asarray(y).astype(bool)
    t_lo, t_hi = np.broadcast_to(t_lo, p.shape), np.broadcast_to(t_hi, p.shape)
    yes, no = p >= t_hi, (p <= t_lo) & ~(p >= t_hi)
    sure = yes | no
    r = {'coverage': float(sure.mean()),
         'acc_sure': float((yes & y).sum() + (no & ~y).sum()) / max(int(sure.sum()), 1),
         'sens_sure': float((yes & y).sum() / max(int((sure & y).sum()), 1)),
         'spec_sure': float((no & ~y).sum() / max(int((sure & ~y).sum()), 1)),
         'missed': int((no & y).sum()), 'n_unsure': int((~sure).sum()),
         'anem_in_unsure': float(y[~sure].mean()) if (~sure).any() else float('nan')}
    return r
