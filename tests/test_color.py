import numpy as np
import pytest

from anemia import color


def patch(rgb, shape=(20, 20)):
    return np.broadcast_to(np.asarray(rgb, np.float32), shape + (3,)).copy()


def test_srgb_roundtrip():
    x = np.linspace(0, 1, 101, dtype=np.float32)
    assert np.allclose(color.lin_to_srgb(color.srgb_to_lin(x)), x, atol=1e-5)
    assert color.srgb_to_lin(np.float32(0.5)) == pytest.approx(0.2140, abs=1e-4)   # стандарт sRGB


def test_erythema_index_known_values():
    assert color.erythema_index(patch([0.8, 0.4, 0.3]))[0, 0] == pytest.approx(np.log10(2), abs=1e-6)
    assert color.erythema_index(patch([0.5, 0.5, 0.5]))[0, 0] == pytest.approx(0, abs=1e-6)
    # бледнее (меньше R относительно G) → ниже EI
    assert color.erythema_index(patch([0.8, 0.6, 0.5]))[0, 0] < color.erythema_index(patch([0.8, 0.4, 0.3]))[0, 0]


def test_erythema_index_invariant_to_exposure_in_srgb_ratio():
    # EI — логарифм отношения, общий множитель яркости сокращается
    a = color.erythema_index(patch([0.6, 0.3, 0.2]))[0, 0]
    b = color.erythema_index(patch([0.3, 0.15, 0.1]))[0, 0]
    assert a == pytest.approx(b, abs=1e-6)


@pytest.mark.parametrize('rgb,hue', [([1, 0, 0], 0), ([0, 1, 0], 120), ([0, 0, 1], 240),
                                     ([1, 1, 0], 60), ([1, 0, 1], 300)])
def test_hsi_hue_primaries(rgb, hue):
    H, S, I = color.rgb_to_hsi(patch(rgb, (1, 1)))
    assert H[0, 0] == pytest.approx(hue, abs=0.01)
    assert S[0, 0] == pytest.approx(1, abs=1e-3)
    assert I[0, 0] == pytest.approx(np.mean(rgb), abs=1e-6)


def test_hsi_gray_has_zero_saturation():
    _, S, I = color.rgb_to_hsi(patch([0.4, 0.4, 0.4], (1, 1)))
    assert S[0, 0] == pytest.approx(0, abs=1e-3) and I[0, 0] == pytest.approx(0.4)


def test_lab_reference_points():
    lab = color.rgb_to_lab(np.array([[[1, 1, 1], [0, 0, 0], [1, 0, 0]]], np.float32))[0]
    assert lab[0] == pytest.approx([100, 0, 0], abs=0.1)
    assert lab[1] == pytest.approx([0, 0, 0], abs=0.1)
    assert lab[2] == pytest.approx([53.24, 80.09, 67.20], abs=0.3)    # sRGB красный, D65


def test_high_hue_ratio():
    img = patch([0.8, 0.3, 0.3])
    img[:10] = [0.3, 0.8, 0.3]
    assert color.high_hue_ratio(img, np.ones(img.shape[:2], bool)) == pytest.approx(0.5)


def test_tissue_mask_drops_glare_and_shadow():
    img = patch([0.7, 0.35, 0.3], (30, 30))
    img[:3] = 1.0          # блик
    img[-3:] = 0.02        # тень
    m = color.tissue_mask(img, np.ones(img.shape[:2], bool))
    assert not m[:3].any() and not m[-3:].any() and m[10:20].all()


def test_roi_features_values():
    f = color.roi_features(patch([0.8, 0.4, 0.3]), np.ones((20, 20), bool), prefix='c_')
    assert f['c_EI_m'] == pytest.approx(np.log10(2), abs=1e-5)
    assert f['c_r_m'] == pytest.approx(0.8 / 1.5, abs=1e-5)
    assert f['c_hue_m'] == pytest.approx(10.9, abs=0.1) and f['c_HHR'] == 0.0     # оттенок 10,9° > 10°
    assert color.roi_features(patch([0.8, 0.3, 0.3]), np.ones((20, 20), bool))['HHR'] == 1.0
    assert color.roi_features(patch([0.8, 0.4, 0.3]), np.zeros((20, 20), bool)) is None


def eye(conj_rgb=(0.75, 0.35, 0.33), sclera_rgb=(0.85, 0.83, 0.80)):
    img = np.zeros((40, 40, 3), np.float32)
    img[:20], img[20:] = conj_rgb, sclera_rgb
    conj = np.zeros((40, 40), bool); conj[:20] = True
    return img, conj, ~conj


def relight_linear(img, gains):
    return color.lin_to_srgb(color.srgb_to_lin(img) * np.asarray(gains, np.float32))


@pytest.mark.parametrize('gains', [(1.3, 1.0, 0.7), (0.8, 1.0, 1.25), (0.6, 0.6, 0.6)])
def test_relative_features_invariant_to_illuminant(gains):
    """Нормализация по склере: множитель источника света (фон Крис) сокращается точно."""
    img, conj, scl = eye()
    a = color.relative_features(img, conj, scl)
    b = color.relative_features(relight_linear(img, gains), conj, scl)
    for k in a:
        assert b[k] == pytest.approx(a[k], abs=0.02), k


def test_white_balance_by_sclera_undoes_illuminant():
    img, conj, scl = eye()
    lit = relight_linear(img, (1.3, 1.0, 0.7))
    raw_shift = np.abs(color.roi_features(lit, conj)['EI_m'] - color.roi_features(img, conj)['EI_m'])
    bal_shift = np.abs(color.roi_features(color.white_balance_by(lit, scl), conj)['EI_m']
                       - color.roi_features(color.white_balance_by(img, scl), conj)['EI_m'])
    assert raw_shift > 0.03 and bal_shift < 0.005
