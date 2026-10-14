import numpy as np
import pytest

from anemia import metrics as M


def test_iou_dice():
    a = np.zeros((10, 10), bool); a[:5] = True
    b = np.zeros((10, 10), bool); b[2:7] = True
    assert M.iou(a, b) == pytest.approx(30 / 70)
    assert M.dice(a, b) == pytest.approx(60 / 100)
    assert M.iou(a, a) == 1.0


def test_wilson_known():
    p, lo, hi = M.wilson(8, 10)
    assert p == 0.8 and lo == pytest.approx(0.490, abs=1e-3) and hi == pytest.approx(0.943, abs=1e-3)


def test_bland_altman():
    r = M.bland_altman([12, 14, 16], [10, 10, 10])
    assert r['bias'] == pytest.approx(4) and r['loa_hi'] - r['loa_lo'] == pytest.approx(2 * 1.96 * 2)


def test_threshold_reaches_target_sensitivity():
    rng = np.random.default_rng(0)
    y = rng.random(500) < 0.4
    s = rng.normal(y.astype(float), 1.0)
    t = M.threshold_for_sensitivity(s, y, 0.8)
    assert (s[y] >= t).mean() >= 0.8
    assert (s[y] >= np.nextafter(t, np.inf)).mean() < 0.8 + 1 / y.sum() + 1e-9   # порог максимальный


def test_bootstrap_ci_contains_estimate():
    rng = np.random.default_rng(1)
    x = rng.normal(5, 1, 200)
    lo, hi = M.bootstrap_ci(np.mean, np.arange(200), 500, 0, x)
    assert lo < x.mean() < hi and hi - lo < 0.5
