"""Качество по исходным пикселям вырезки (anemia/crop_quality.py): резкость без увеличения, реальные пиксели века,
шум и блоки JPEG. Синтетика — проверка кода, не точности."""
import io

import cv2
import numpy as np
import pytest

from anemia import crop_quality as CQ
from anemia import regions as R


def texture(h, w, seed=0):
    """Резкая «ткань»: случайные цветные круги с чёткими краями (без сетки, чтобы не путать с блоками JPEG)."""
    rng = np.random.default_rng(seed)
    big = np.full((h * 4, w * 4, 3), 0.6, np.float32)
    for _ in range(h * w // 40 + 20):
        c = tuple(float(v) for v in rng.uniform(0.2, 0.9, 3))
        cv2.circle(big, (int(rng.integers(0, w * 4)), int(rng.integers(0, h * 4))), int(rng.integers(8, 40)), c, -1)
    return cv2.resize(big, (w, h), interpolation=cv2.INTER_AREA)    # резкое уменьшение, как у камеры


def lid_mask(h, w):
    m = np.zeros((h, w), bool)
    m[int(0.55 * h):int(0.75 * h), int(0.15 * w):int(0.85 * w)] = True
    return m


def jpeg(img, q):
    _, b = cv2.imencode('.jpg', (np.clip(img, 0, 1)[..., ::-1] * 255 + 0.5).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, q])
    return cv2.imdecode(b, 1)[..., ::-1].astype(np.float32) / 255


def test_never_upscales_for_measurement():
    assert CQ.measure_scale((60, 80)) == 1.0
    assert CQ.measure_scale((640, 480)) == 0.5


def test_small_sharp_eye_is_not_called_blurred():
    src = texture(40, 54)                                       # мелкая, но резкая вырезка: веко 38 px
    m = lid_mask(40, 54)
    up = R._prepare(src)                                        # как раньше: увеличение до 320 px
    mu = cv2.resize(m.astype(np.uint8), (up.shape[1], up.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    assert R.blur_ratio(up, mu) < CQ.BLUR_MIN                   # старое измерение: «размыт» из-за увеличения
    q = CQ.measure(src, m)
    assert q['blur'] >= CQ.BLUR_MIN                             # по исходным пикселям — резкий
    assert not any('размыт' in x for x in CQ.check(q))


def test_large_defocused_eye_still_blurred_and_check_not_weaker():
    src = cv2.GaussianBlur(texture(960, 1280), (0, 0), 14)      # крупная, но расфокусированная
    m = lid_mask(960, 1280)
    q = CQ.measure(src, m)
    assert any('размыт' in x for x in CQ.check(q))
    up = R._prepare(src)                                        # крупная вырезка: то же измерение, что и раньше
    mu = cv2.resize(m.astype(np.uint8), (up.shape[1], up.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    assert abs(q['blur'] - R.blur_ratio(up, mu)) < 0.05


def test_small_but_blurred_eye_is_blurred():
    src = cv2.GaussianBlur(texture(120, 160), (0, 0), 2)        # пикселей хватает, но расфокус ~1,7% ширины века
    q = CQ.measure(src, lid_mask(120, 160))
    assert q['blur'] >= CQ.BLUR_MIN                             # абсолютный порог не ловит
    assert any('размыт относительно размера глаза' in x for x in CQ.check(q))
    sharp = CQ.measure(texture(120, 160), lid_mask(120, 160))
    assert CQ.check(sharp) == []


def test_lid_pixels_measured_on_source_not_on_crop():
    src = texture(90, 120)
    m320 = cv2.resize(lid_mask(90, 120).astype(np.uint8), (320, 240), interpolation=cv2.INTER_NEAREST) > 0
    q = CQ.measure(src, m320)                                   # маска на увеличенной вырезке → px исходника
    assert abs(q['lid_w'] - 0.7 * 120) <= 2
    assert abs(q['lid_h'] - 0.2 * 90) <= 2


def test_resolution_threshold_per_artifact_level():
    lim = CQ.LIMITS
    clean = {'lid_w': lim['lid_w_clean'], 'lid_h': 10, 'blur': 5.0, 'measure_scale': 1.0, 'noise': 0.0, 'block': 1.0}
    assert CQ.check(clean, file_q='lossless') == []
    assert any('мало реальных пикселей' in x for x in CQ.check(dict(clean, lid_w=lim['lid_w_clean'] - 1), file_q='lossless'))
    art = dict(clean, block=lim['block_max'] + 0.1)
    assert lim['lid_w_artifact'] > lim['lid_w_clean']           # при артефактах сжатия деталей меньше — порог строже
    assert any('блоки JPEG' in x for x in CQ.check(art, file_q='lossless'))
    assert CQ.check(dict(art, lid_w=lim['lid_w_artifact']), file_q='lossless') == []
    assert CQ.check(dict(clean, lid_w=1000, blur=1.2))          # абсолютное размытие — отказ при любом размере


def test_needed_lid_pixels_by_jpeg_quality_never_weaker():
    q = {'lid_w': 0, 'blur': 5.0, 'measure_scale': 1.0, 'block': 1.0}
    need = {fq: CQ.lid_needed(q, file_q=fq)[0] for fq in ('lossless', 100, 95, 92, 90, 85, 75, 40, None)}
    assert need['lossless'] == CQ.LIMITS['lid_w_clean']
    assert need[100] == need[95] <= need[92] == need[90] <= need[85] <= need[75] == need[40] == CQ.LIMITS['lid_w_artifact']
    assert need[None] == CQ.LIMITS['lid_w_unknown'] >= need[95]  # неизвестное сжатие — не мягче JPEG 95
    blocky = dict(q, block=CQ.LIMITS['block_max'] + 0.2)
    assert all(CQ.lid_needed(blocky, file_q=fq)[0] >= CQ.LIMITS['lid_w_artifact'] for fq in ('lossless', 95, None))


@pytest.mark.parametrize('q', [98, 95, 90, 85, 75, 60, 30])
def test_jpeg_quality_from_file_tables(q, tmp_path):
    img = (texture(64, 64) * 255).astype(np.uint8)
    data = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()
    assert abs(CQ.jpeg_quality(data) - q) <= 1
    path = tmp_path / 'a.jpg'
    path.write_bytes(data)
    assert abs(CQ.jpeg_quality(str(path)) - q) <= 1
    f = io.BytesIO(data)
    f.seek(0)
    assert abs(CQ.jpeg_quality(f) - q) <= 1 and f.tell() == 0      # файловый объект не «съеден»


def test_jpeg_quality_unknown_for_png_and_arrays():
    img = (texture(32, 32) * 255).astype(np.uint8)
    assert CQ.jpeg_quality(cv2.imencode('.png', img)[1].tobytes()) is None
    assert CQ.jpeg_quality(img) is None
    assert CQ.jpeg_quality(b'not an image') is None


def test_jpeg_blocks_detected_on_photo_grid():
    src = texture(160, 240, seed=3)
    src = cv2.GaussianBlur(src, (0, 0), 1.2)
    clean = CQ.blockiness(src)
    comp = jpeg(src, 20)
    assert CQ.blockiness(comp) > clean + 0.1
    crop = comp[13:120, 21:200]                                 # вырезка не с начала фото: сетка — по фото
    assert CQ.blockiness(crop, offset=(21, 13)) > CQ.blockiness(crop, offset=(0, 0))


def test_noise_estimate_grows_with_noise():
    flat = np.full((100, 100, 3), 0.5, np.float32)
    rng = np.random.default_rng(0)
    noisy = np.clip(flat + 0.03 * rng.standard_normal(flat.shape), 0, 1).astype(np.float32)
    assert CQ.noise_sigma(flat) < 0.1 and CQ.noise_sigma(noisy) > 3


def test_fit_limits_rule():
    rows = []
    for w in (20, 30, 40, 60, 80, 120, 200):
        for _ in range(30):
            base = {'iou_drop': 0.0, 'blur': 6.0, 'measure_scale': 1.0}
            rows.append({**base, 'kind': 'clean', 'lid_w': w, 'drift': 0.05 if w >= 40 else 0.3, 'block': 1.0})
            rows.append({**base, 'kind': 'artifact', 'lid_w': w, 'drift': 0.05 if w >= 80 else 0.3, 'block': 1.4})
        for b in (1.2, 1.5, 2.0, 3.0):                           # расфокус: хуже при меньшей резкости
            rows += [{'kind': 'defocus', 'lid_w': 100, 'drift': 0.05 if b >= 2.0 else 0.4, 'iou_drop': 0.0, 'blur': b,
                      'measure_scale': 1.0, 'block': 1.0}] * 30
    lim = CQ.fit_limits(rows)
    assert 38 <= lim['lid_w_clean'] <= 42 and 78 <= lim['lid_w_artifact'] <= 82
    assert 1.0 <= lim['block_max'] < 1.4
    assert CQ.rel_blur({'lid_w': 100, 'blur': 2.0, 'measure_scale': 1.0}, lim) <= lim['rel_blur_max']
    assert CQ.rel_blur({'lid_w': 100, 'blur': 1.5, 'measure_scale': 1.0}, lim) > lim['rel_blur_max']
