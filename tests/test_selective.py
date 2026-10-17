"""Зона «не уверен»: пороги по обучающим данным дают заданную чистоту уверенных ответов. Синтетика — проверка кода."""
import numpy as np

from anemia.selective import zone_metrics, zone_thresholds


def test_zone_thresholds_reach_target_on_train():
    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.4
    p = np.clip(0.5 + (y - 0.5) * 0.6 + rng.normal(0, 0.25, 400), 0, 1)
    t_lo, t_hi = zone_thresholds(p, y, 0.97)
    assert t_lo < t_hi
    z = zone_metrics(p, y, t_lo, t_hi)
    assert z['acc_sure'] >= 0.97 and 0 < z['coverage'] < 1
    assert z['n_unsure'] == int(((p > t_lo) & (p < t_hi)).sum())


def test_zone_thresholds_inseparable_collapse():
    y = np.array([0, 1] * 20, bool)
    p = np.full(40, 0.5)
    t_lo, t_hi = zone_thresholds(p, y, 0.90)
    assert np.isinf(t_lo) or t_lo == t_hi
