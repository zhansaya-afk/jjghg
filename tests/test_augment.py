import numpy as np
import pytest

from anemia.augment import STRESS_CONDITIONS, apply_illuminant, kelvin_gains, planck_xy


def test_reference_is_identity():
    assert kelvin_gains(6500) == pytest.approx([1, 1, 1], abs=1e-6)


def test_planck_xy_known_points():
    assert planck_xy(2856) == pytest.approx((0.4476, 0.4074), abs=2e-3)     # иллюминант A
    assert planck_xy(6500) == pytest.approx((0.3135, 0.3236), abs=2e-3)


def test_warm_light_is_red_cold_is_blue():
    w, c = kelvin_gains(2700), kelvin_gains(7500)
    assert w[0] > 1 > w[2] and c[2] > 1 > c[0]


def test_apply_illuminant_shape_and_range():
    img = np.random.default_rng(0).random((8, 8, 3)).astype(np.float32)
    for T, e in STRESS_CONDITIONS:
        out = apply_illuminant(img, T, e)
        assert out.shape == img.shape and out.min() >= 0 and out.max() <= 1
