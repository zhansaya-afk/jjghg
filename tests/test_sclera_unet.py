"""Теневой режим U-Net склеры: формат, совпадение с sclera_kit, отсутствие влияния на fingerprint."""

import os

import numpy as np
import pytest

from anemia import sclera_unet as U
from anemia.model_store import FEATURE_MODULES, code_version

MODEL_DIR = os.environ.get("SCLERA_MODEL_DIR", U.model_dir())


def test_not_in_fingerprint():
    assert "sclera_unet.py" not in FEATURE_MODULES
    assert len(code_version()) == 12


def test_pipeline_does_not_use_shadow():
    import anemia

    root = os.path.dirname(anemia.__file__)
    for name in os.listdir(root):
        if name.endswith(".py") and name != "sclera_unet.py":
            with open(os.path.join(root, name), encoding="utf-8") as f:
                assert "sclera_unet" not in f.read(), name


def test_clean_keeps_main_and_fills_holes():
    x = np.full((200, 200, 3), 0.9, np.float32)
    m = np.zeros((200, 200), bool)
    m[50:150, 50:150] = True
    m[95:105, 95:105] = False  # дырка (сосуд) — заливается
    m[5:12, 5:12] = True  # мелкое пятно — убирается
    out = U.clean(x, m)
    assert out[100, 100] and not out[8, 8] and out.dtype == bool


def test_shadow_without_crop():
    assert (
        U.shadow({"crop": None}, np.zeros((10, 10, 3), np.float32), model=None) is None
    )


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("torch")
    pytest.importorskip("segmentation_models_pytorch")
    if not os.path.exists(os.path.join(MODEL_DIR, "best.pt")):
        pytest.skip(f"нет весов {MODEL_DIR}")
    return U.ScleraModel(MODEL_DIR)


def test_reference_output(model):
    p = model.prob_small(U.reference_input())
    assert abs(p.mean() - 0.221169) < 2e-3 and abs(int((p > 0.5).sum()) - 8603) < 60


def test_shadow_routes(model):
    from anemia import regions as R

    rng = np.random.default_rng(0)
    rgb = np.clip(
        U.reference_input()[::4, ::4] + rng.normal(0, 0.01, (996, 747, 3)), 0, 1
    ).astype(np.float32)
    crop = U._closeup_frame(rgb)
    sh = U.shadow({"crop": crop, "sclera": np.zeros(crop.shape[:2], bool)}, rgb, model)
    assert (
        sh["validated"]
        and sh["mask"].shape == crop.shape[:2]
        and sh["mask"].dtype == bool
    )
    box = R._prepare(rgb[200:700, 100:600])
    sh2 = U.shadow({"crop": box, "sclera": None}, rgb, model)
    assert not sh2["validated"] and sh2["mask"].shape == box.shape[:2]
    assert U.compare_overlay({"crop": crop, "sclera": sh["mask"]}, sh).dtype == np.uint8


# ---- шаг 1 (ограничения, TTA): только проверка кода на синтетических массивах, не точность ----


class _Fake(U.ScleraModel):
    """«Сеть» = красный канал входа, сжатый как в prob_small; без torch и весов."""

    def __init__(self):
        self.cfg = {"thr": 0.5}

    def prob_small(self, x):
        import cv2

        w, h = (288, 224) if x.shape[1] >= x.shape[0] else (224, 288)
        return cv2.resize(
            x[..., 0].astype(np.float32), (w, h), interpolation=cv2.INTER_AREA
        )


def _spot(h, w):
    x = np.zeros((h, w, 3), np.float32)
    x[: h // 3, : w // 4, 0] = 1  # асимметричное пятно
    return x


@pytest.mark.parametrize("hw", [(640, 480), (480, 640), (513, 384)])
def test_tta_inverse_and_id_equals_predict(hw):
    x = _spot(*hw)
    m = _Fake()
    one, used = U.predict_tta(m, x, post="raw", hflip=False, landscape=False)
    assert used == ["id"] and np.array_equal(one, m.predict(x, post="raw"))
    p0 = m.prob_small(x)
    for name, fwd, inv in U._views(
        *hw
    ):  # каждый вид после обратного преобразования = id
        assert np.allclose(
            np.ascontiguousarray(inv(m.prob_small(fwd(x)))), p0, atol=1e-6
        ), name
    assert set(U.check_views(*hw)) == {n for n, _, _ in U._views(*hw)}


def test_lid_region_per_component():
    c = np.zeros((100, 100), bool)
    c[70:86, 30:90] = True  # нижнее веко
    c[10:14, 0:12] = True  # отдельный кусок слева
    lid = U.lid_region(c)
    assert lid[95, 50] and not lid[60, 50]  # ниже / выше края века
    assert not lid[50, 20]  # между частями не интерполируется
    with pytest.raises(ValueError):
        U.lid_region(None)


def test_constrain_and_lid_only_below_iris():
    c = np.zeros((100, 100), bool)
    c[70:80, 20:80] = True
    m = np.ones((100, 100), bool)
    out = U.constrain(m, c, (50, 40, 10))
    assert not out[75, 50] and not out[90, 50] and not out[40, 50] and out[20, 50]
    out_up = U.constrain(
        m, c, (50, 95, 2)
    )  # «веко» выше радужки: ниже края не вычитается
    assert out_up[88, 50] and not out_up[75, 50]


def test_shadow_limits_skip_unchecked_conj(monkeypatch):
    rgb = np.full((60, 80, 3), 0.8, np.float32)
    crop = np.full((30, 40, 3), 0.8, np.float32)
    conj = np.zeros((30, 40), bool)
    conj[20:25, 5:35] = True
    m = _Fake()
    monkeypatch.setattr(U, "_closeup_frame", lambda x: np.zeros((1, 1, 3), np.float32))
    monkeypatch.setattr(
        m, "predict", lambda x, post=None: np.ones(x.shape[:2], bool), raising=False
    )
    r = {"crop": crop, "conj": conj, "sclera": None, "tissues": []}
    sh = U.shadow(r, rgb, m, limits=True)
    assert sh["limits_note"] and sh["removed_px"]["conj"] == 0 and sh["mask"][22, 20]
    sh2 = U.shadow(dict(r, tissues=["conj"]), rgb, m, limits=True)
    assert sh2["limits_note"] is None and not sh2["mask"][22, 20]
    assert U.shadow(r, rgb, m)["mask"].all()  # по умолчанию ничего не вычитается
