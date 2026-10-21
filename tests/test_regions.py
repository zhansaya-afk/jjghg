"""Поиск конъюнктивы и склеры (anemia/regions.py) на синтетических снимках с известной разметкой.
Проверяется логика режимов и проверок правдоподобия; точность на реальных фото меряет ноутбук (Dice с врачами)."""
import cv2
import numpy as np
import pytest

from anemia import regions

SKIN, SCLERA, IRIS, CONJ = (0.80, 0.62, 0.50), (0.93, 0.92, 0.90), (0.18, 0.12, 0.10), (0.85, 0.38, 0.40)


def _noise(img, seed=0):
    rng = np.random.default_rng(seed)
    return np.clip(img + rng.normal(0, 0.012, img.shape), 0, 1).astype(np.float32)


def synthetic_eye(h=300, w=400, with_iris=True, with_sclera=True, with_conj=True):
    """Кожа; белая «миндалина» склеры; тёмная радужка; розовая полоса вывернутого века под склерой."""
    img = np.empty((h, w, 3), np.float32)
    img[:] = SKIN
    sclera = np.zeros((h, w), np.uint8)
    conj = np.zeros((h, w), np.uint8)
    cx, cy, r = w // 2, int(h * 0.42), 42
    if with_conj:
        cv2.ellipse(conj, (cx, cy + 50), (115, 45), 0, 0, 180, 1, -1)
        cv2.ellipse(conj, (cx, cy + 50), (115, 12), 0, 0, 180, 0, -1)
    if with_sclera:
        cv2.ellipse(sclera, (cx, cy), (112, 62), 0, 0, 360, 1, -1)    # глазная щель ≈ 2,5 диаметра радужки
    img[conj > 0] = CONJ
    img[sclera > 0] = SCLERA
    iris = np.zeros((h, w), np.uint8)
    if with_iris:
        cv2.circle(iris, (cx, cy), r, 1, -1)
        img[iris > 0] = IRIS
    return _noise(img), sclera.astype(bool) & ~iris.astype(bool), conj.astype(bool) & ~sclera.astype(bool)


def _iou(a, b):
    return (a & b).sum() / max((a | b).sum(), 1)


def _resized(mask, like):
    return cv2.resize(mask.astype(np.uint8), (like.shape[1], like.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def test_full_eye_finds_both_tissues():
    img, sc, cj = synthetic_eye()
    r = regions.detect_regions(img, 'auto')
    assert r['use'] == {'conj', 'sclera'}
    assert r['iris'] is not None
    assert _iou(r['conj'], _resized(cj, r['conj'])) > 0.6
    assert _iou(r['sclera'], _resized(sc, r['sclera'])) > 0.6
    assert not (r['conj'] & r['sclera']).any()


@pytest.mark.parametrize('mode,expected', [('conj', {'conj'}), ('sclera', {'sclera'}), ('both', {'conj', 'sclera'})])
def test_modes_restrict_tissues(mode, expected):
    img, _, _ = synthetic_eye()
    r = regions.detect_regions(img, mode)
    assert r['use'] == expected
    if 'sclera' not in expected:
        assert not r['sclera'].any()
    if 'conj' not in expected:
        assert not r['conj'].any()


def test_conj_closeup_without_iris_is_not_rejected():
    """Главный баг v11: крупный план века без радужки → пустая маска → «переснять»."""
    img, _, cj = synthetic_eye(with_iris=False, with_sclera=False)
    r = regions.detect_regions(img, 'auto')
    assert 'conj' in r['use']
    assert _iou(r['conj'], _resized(cj, r['conj'])) > 0.6
    assert regions.quality(r['crop'], r['conj'], r['sclera'], 1000) == []


def test_cutout_conj():
    img = np.zeros((200, 300, 3), np.float32)
    m = np.zeros((200, 300), np.uint8)
    cv2.ellipse(m, (150, 60), (130, 110), 0, 20, 160, 1, -1)
    img[m > 0] = CONJ
    img = _noise(img) * (m > 0)[..., None]
    r = regions.detect_regions(img, 'auto')
    assert r['kind'] == 'вырезанная ткань' and r['use'] == {'conj'}
    # пользователь ошибся режимом: красная ткань не принимается за склеру
    assert regions.detect_regions(img, 'sclera')['use'] == set()


def test_skin_only_finds_nothing():
    img = _noise(np.broadcast_to(np.float32(SKIN), (240, 320, 3)).copy())
    r = regions.detect_regions(img, 'auto')
    assert r['use'] == set()
    assert 'не видно ни конъюнктивы, ни склеры' in regions.quality(r['crop'], r['conj'], r['sclera'], 1000)


def test_both_mode_falls_back_to_one_tissue():
    img, _, _ = synthetic_eye(with_conj=False)
    r = regions.detect_regions(img, 'both')
    assert r['use'] == {'sclera'}
    assert 'оценка по одной ткани' in r['notes']


def test_blur_is_detected_and_smooth_tissue_is_not_blur():
    img, _, _ = synthetic_eye()
    r = regions.detect_regions(img, 'auto')
    roi = r['conj'] | r['sclera']
    assert regions.blur_ratio(r['crop'], roi) > 1.5
    blurred = cv2.GaussianBlur(r['crop'], (0, 0), 3)
    assert regions.blur_ratio(blurred, roi) < 1.3


def test_bad_mode_raises():
    with pytest.raises(ValueError):
        regions.detect_regions(np.zeros((10, 10, 3), np.float32), 'eye')


class _FakeLandmarker:
    """Подмена MediaPipe: возвращает 478 точек лица, где контур правого глаза и радужка — там, где мы их нарисовали."""

    def __init__(self, eye_box, iris, size):
        from types import SimpleNamespace as NS
        (x0, y0, x1, y1), (cx, cy, r), (h, w) = eye_box, iris, size
        pts = [NS(x=0.5, y=0.5) for _ in range(478)]
        for k, i in enumerate(regions.R_EYE):
            t = 2 * np.pi * k / len(regions.R_EYE)
            pts[i] = NS(x=((x0 + x1) / 2 + (x1 - x0) / 2 * np.cos(t)) / w, y=((y0 + y1) / 2 + (y1 - y0) / 2 * np.sin(t)) / h)
        for k, i in enumerate(regions.L_EYE):                     # второй глаз — на пустой коже справа
            pts[i] = NS(x=(0.85 * w + 20 * np.cos(k)) / w, y=(0.2 * h + 8 * np.sin(k)) / h)
        pts[regions.R_IRIS[0]] = NS(x=cx / w, y=cy / h)
        for k, i in enumerate(regions.R_IRIS[1:]):
            pts[i] = NS(x=(cx + r * np.cos(k * np.pi / 2)) / w, y=(cy + r * np.sin(k * np.pi / 2)) / h)
        self.res = NS(face_landmarks=[pts])

    def detect(self, image):
        return self.res


def test_face_photo_uses_eye_not_lips():
    """Фото лица: ткани ищутся только в вырезке глаза; красные губы не становятся «конъюнктивой»."""
    pytest.importorskip('mediapipe')
    eye, _, cj = synthetic_eye()
    H, W = 900, 800
    face = np.empty((H, W, 3), np.float32)
    face[:] = SKIN
    face[100:400, 200:600] = eye
    cv2.ellipse(face, (400, 750), (160, 50), 0, 0, 360, (0.80, 0.25, 0.30), -1)   # губы
    face = _noise(face, 1)
    fl = _FakeLandmarker((250, 165, 550, 215), (400, 226, 42), (H, W))
    r = regions.detect_regions(face, 'auto', landmarker=fl)
    assert r['kind'].startswith('лицо')
    assert 'conj' in r['use'] and 'sclera' in r['use']
