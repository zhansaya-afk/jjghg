"""Контроль качества фото: размытие, темнота, пересвет, блик, не тот кадр — и подсказки «переснимите, потому что …»
(anemia/retake.py). Синтетический глаз и модель из tests/test_core — проверка кода, не точности."""
import cv2
import numpy as np
import pytest

from anemia import api as API
from anemia import crop_quality as CQ
from anemia import predict as PR
from anemia import regions as R
from anemia import retake as RT
from test_core import _bundle
from test_regions import synthetic_eye


@pytest.fixture(scope='module')
def bundle():
    return _bundle()


@pytest.fixture(scope='module')
def eye():
    img, _, conj = synthetic_eye()
    return img, conj > 0


def camera(x, sigma=2 / 255, seed=0):
    rng = np.random.default_rng(seed)
    x = np.clip(np.asarray(x, np.float32) + rng.normal(0, sigma, x.shape).astype(np.float32), 0, 1)
    return np.round(x * 255).astype(np.uint8)


def analyze(img, bundle):
    return PR.analyze_photo(img, sex='Ж', age=30, bundle=bundle, landmarker=None)


def codes(r):
    return {h['code'] for h in RT.hints(r)}


def test_noise_fools_laplacian_blur_ratio_but_not_edge_sigma(eye):
    """Ошибка до 2026-11-18: шум и 8 бит делают сильно размытый снимок «резким» для blur_ratio."""
    img, m = eye
    sharp = camera(img).astype(np.float32) / 255
    blurred = camera(cv2.GaussianBlur(img, (0, 0), 10)).astype(np.float32) / 255
    assert R.blur_ratio(blurred, m) > 1.3                       # старая мера: «резкий»
    assert CQ.edge_sigma(blurred, m) > CQ.edge_sigma(sharp, m) + 2
    side = max(img.shape[:2])
    assert CQ.edge_sigma(blurred, m) / side > CQ.EDGE_SIGMA_REL_MAX
    assert CQ.edge_sigma(sharp, m) / side < CQ.EDGE_SIGMA_REL_MAX


def test_edge_sigma_ignores_noise(eye):
    img, m = eye
    x = cv2.GaussianBlur(img, (0, 0), 6)
    a = CQ.edge_sigma(camera(x, 0).astype(np.float32) / 255, m)
    b = CQ.edge_sigma(camera(x, 8 / 255).astype(np.float32) / 255, m)
    assert abs(a - b) < 0.5


def test_edge_sigma_empty_roi_is_nan(eye):
    img, m = eye
    assert np.isnan(CQ.edge_sigma(img, np.zeros_like(m)))


def test_clean_photo_not_refused_by_quality(eye, bundle):
    r = analyze(camera(eye[0]), bundle)
    assert r['status'] != 'retake'


def test_heavy_blur_with_noise_refused_with_blur_hint(eye, bundle):
    r = analyze(camera(cv2.GaussianBlur(eye[0], (0, 0), 10)), bundle)
    assert r['status'] == 'retake' and r['stage'] == 'quality'
    assert 'blur' in codes(r)
    assert RT.retake_text(r).startswith('Переснимите, потому что снимок размыт')


@pytest.mark.parametrize('k, code', [(0.08, 'dark'), (4.0, 'overexposed')])
def test_exposure(eye, bundle, k, code):
    r = analyze(camera(np.clip(eye[0] * k, 0, 1)), bundle)
    assert r['status'] == 'retake' and code in codes(r)


def test_big_glare_on_lid_refused(eye, bundle):
    img, m = eye
    g = img.copy()
    g[m] = 1.0                                                  # блик закрыл всё веко
    r = analyze(camera(g), bundle)
    assert r['status'] == 'retake' and r['risk'] is None and RT.hints(r)


@pytest.mark.parametrize('kind, expected', [('gray', {'no_color'}), ('skin', {'no_eye'}), ('document', {'no_color'}),
                                            ('tiny', {'frame_small'})])
def test_wrong_frame(eye, bundle, kind, expected):
    img = eye[0]
    rng = np.random.default_rng(1)
    x = {'gray': np.repeat(img.mean(-1, keepdims=True), 3, -1),
         'skin': np.full_like(img, (0.8, 0.6, 0.5)) + rng.normal(0, 0.02, img.shape),
         'document': np.full_like(img, 0.95),
         'tiny': img[:20, :20]}[kind]
    r = analyze(camera(x), bundle)
    assert r['status'] == 'retake' and r['risk'] is None and r['hb_range'] is None
    assert expected <= codes(r)


@pytest.mark.parametrize('issue, code', [
    ('изображение слишком маленькое (20×20); нужно хотя бы 64 px по короткой стороне', 'frame_small'),
    ('изображение почти без цвета (чёрно-белое или сильный фильтр)', 'no_color'),
    ('кадр почти чёрный — включите свет', 'dark'), ('слишком темно', 'dark'),
    ('кадр пересвечен — уберите вспышку или прямой свет', 'overexposed'), ('пересвет / вспышка', 'overexposed'),
    ('снимок размыт (резкость 1.21 < 1.3 на исходных пикселях, без увеличения)', 'blur'),
    ('мало реальных пикселей: веко 40 px на исходном фото', 'small'),
    ('ткань слишком мелкая в кадре — снимите ближе', 'small'),
    ('глаз не найден — снимите глаз с оттянутым нижним веком', 'no_eye'),
    ('конъюнктива: не найдена: конъюнктива не видна или слишком мала', 'no_tissue'),
    ('это уже вырезанная ткань, а модель взрослых обучена на снимках глаза целиком', 'cutout'),
    ('мало чистой ткани для признаков цвета (блики, тени, ресницы)', 'few_px'),
    ('60% признаков цвета далеко вне диапазона обучающих снимков', 'ood'),
    ('что-то новое', 'other')])
def test_classify(issue, code):
    assert RT.classify(issue) == code


def test_no_hint_when_ok_age_or_no_model():
    for r in ({'status': 'ok', 'issues': []}, {'status': 'unreliable', 'stage': 'age', 'issues': ['младше 5 лет']},
              {'status': 'unreliable', 'stage': 'no_model', 'issues': ['нет модели']}):
        assert RT.hints(r) == [] and RT.retake_text(r) == ''


def test_hints_dedup_and_order():
    r = {'status': 'retake', 'stage': 'quality',
         'issues': ['снимок размыт', 'конъюнктива: плохое качество: снимок размыт', 'слишком темно']}
    h = RT.hints(r)
    assert [x['code'] for x in h] == ['dark', 'blur']
    t = RT.retake_text(r)
    assert t.startswith('Переснимите, потому что слишком темно') and '1)' in t and '2)' in t


def test_no_eye_drops_tissue_hint():
    r = {'status': 'retake', 'stage': 'no_eye', 'issues': ['глаз не найден', 'конъюнктива: не найдена']}
    assert [x['code'] for x in RT.hints(r)] == ['no_eye']


def test_hint_in_text_and_api(eye, bundle):
    r = analyze(camera(np.clip(eye[0] * 0.08, 0, 1)), bundle)
    assert 'Переснимите, потому что' in PR.format_result(r)
    res = API.to_result(r, 'test')
    assert res.retake.startswith('Переснимите, потому что') and res.to_dict(images=False)['retake'] == res.retake


def test_qc_stress_script_runs(bundle):
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'qc_stress.py'
    spec = importlib.util.spec_from_file_location('qc_stress', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    img, _, _ = synthetic_eye()
    T = m.run({'eye': img}, bundle)
    assert set(T['вариант']) == set(m.EXPECT)
    S = m.summary(T)
    assert S.iloc[-1]['вариант'] == 'ВСЕ ПЛОХИЕ' and S.iloc[-1]['n'] == len(m.EXPECT) - 1
    for v in ('dark', 'gray', 'document', 'blur_3'):
        assert T.set_index('вариант').loc[v, 'status'] == 'retake'
