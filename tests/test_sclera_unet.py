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
