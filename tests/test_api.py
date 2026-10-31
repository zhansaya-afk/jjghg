"""Единый API (anemia/api.py): screen() и load_models(). Синтетика и малая модель — проверяется контракт, не точность."""
import base64
import io
import json

import joblib
import numpy as np
import pytest
from PIL import Image
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from test_core import _bundle, _fake_det, _synthetic
from test_regions import synthetic_eye

from anemia import api as API
from anemia import core as CORE
from anemia import model_store as STORE
from anemia import predict as PR
from anemia import v12_model as V12


def _jpeg(img):
    buf = io.BytesIO()
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(buf, format='JPEG', quality=95)
    return buf.getvalue()


def _save(bundle, tmp_path):
    bundle = {**bundle, 'compat': STORE.compat_stamp(), 'commit': 'abc1234'}
    path = tmp_path / 'hemavision_core.joblib'
    joblib.dump(bundle, path)
    return path


@pytest.fixture(scope='module')
def models(tmp_path_factory):
    """Одна малая модель на модуль (как small_models в test_core): обе ткани + только склера."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(V12, '_clf', lambda: HistGradientBoostingClassifier(max_iter=10, random_state=0))
        mp.setattr(CORE, '_hb_model', lambda: HistGradientBoostingRegressor(max_iter=10, random_state=0))
        B = _bundle()
        F, _ = _synthetic()
        F = F.rename(columns={'c_x': 'c_EI_m', 'c_z': 'c_a_m', 's_x': 's_EI_m'})
        S = CORE.usable(F, 'eyes_defy_sclera')
        B['models']['eyes_defy_sclera'] = CORE.fit_core(S, ['s_EI_m', 'male'], 'eyes_defy_sclera')
    return API.load_models(_save(B, tmp_path_factory.mktemp('model')))


def _route(monkeypatch, use, eye='весь кадр'):
    """Ткани «найдены» заданные, OOD выключен (маршрут и перевод ответа; сегментация — test_eye_location)."""
    monkeypatch.setattr(PR, 'quality', lambda *a: [])
    monkeypatch.setattr(PR, '_detect', lambda rgb, lm=None: _fake_det(use, eye=eye))
    monkeypatch.setattr(PR, 'learned_conj', lambda det, seg, **kw: det)
    monkeypatch.setattr(CORE, 'ood_share', lambda x, lo, hi: np.zeros(len(x)))  # малая модель учена не на этих признаках


def _undetermined(res):
    assert res.risk == API.RISK_UNKNOWN and res.status != 'ok'
    assert res.p_anemia is None and res.hb_interval_g_l is None and res.tissue is None
    assert res.reason


# ------------------------------------------------------------------ нормальное фото
def test_normal_photo(models, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'})
    res = API.screen(_jpeg(synthetic_eye()[0]), 'F', 30, None, models=models)
    assert res.status == 'ok' and res.risk in CORE.RISK_LEVELS
    assert 0 <= res.p_anemia <= 1 and res.tissue == 'обе'
    assert 'конъюнктивы и склеры' in res.reason
    assert res.quality == {'sharpness': 'ok', 'light': 'ok', 'tissue_visible': {'conjunctiva': True, 'sclera': True}}
    assert res.masks['conjunctiva'].any() and res.masks['sclera'].any()
    assert res.masks['conjunctiva'].shape == res.crop.shape[:2] and res.crop.dtype == np.uint8
    hb = res.hb_interval_g_l
    assert hb is None or (len(hb) == 2 and hb[0] < hb[1])
    assert res.model_version.startswith('core-') and res.model_version.endswith('+abc1234')
    d = json.loads(res.to_json())
    assert d['risk'] == res.risk and d['tissue'] == 'обе'
    m = Image.open(io.BytesIO(base64.b64decode(d['masks']['conjunctiva'])))
    assert np.array_equal(np.asarray(m) > 0, res.masks['conjunctiva'])
    assert 'masks' not in res.to_dict(images=False)


def test_normal_photo_without_mocks_does_not_raise(models):
    """Весь путь ядра на синтетике: любой исход, но без исключения и без цифр при «не определён»."""
    res = API.screen(synthetic_eye()[0], 'М', 45, False, models=models)
    if res.risk == API.RISK_UNKNOWN:
        _undetermined(res)
    else:
        assert res.risk in CORE.RISK_LEVELS and res.tissue in ('обе', 'конъюнктива', 'склера')
    json.loads(res.to_json())


def test_symptoms_pass_through_and_do_not_change_risk(models, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'})
    img = synthetic_eye()[0]
    a = API.screen(img, 'F', 30, None, models=models)
    b = API.screen(img, 'F', 30, None, symptoms=['слабость', 'одышка'], models=models)
    c = API.screen(img, 'F', 30, None, symptoms={'слабость': True, 'головокружение': 0}, models=models)
    assert a.p_anemia == b.p_anemia == c.p_anemia and a.risk == b.risk == c.risk
    assert b.symptoms == {'слабость': True, 'одышка': True} and c.symptoms == {'слабость': True, 'головокружение': 0.0}
    assert API.SYMPTOMS_NOTE in b.warnings and API.SYMPTOMS_NOTE not in a.warnings


# ------------------------------------------------------------------ только склера
def test_sclera_only(models, monkeypatch):
    _route(monkeypatch, {'sclera'})
    res = API.screen(_jpeg(synthetic_eye()[0]), 'M', 50, None, models=models)
    assert res.status == 'ok' and res.tissue == 'склера', (res.stage, res.reason)
    assert 'только по цвету склеры' in res.reason and 'конъюнктива: не найдена' in res.reason
    assert res.quality['tissue_visible'] == {'conjunctiva': False, 'sclera': True}
    assert not res.masks['conjunctiva'].any() and res.masks['sclera'].any()


def test_atypical_photo_unreliable(models, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'})
    monkeypatch.setattr(CORE, 'ood_share', lambda x, lo, hi: np.ones(len(x)))
    res = API.screen(synthetic_eye()[0], 'F', 30, None, models=models)
    _undetermined(res)
    assert res.status == 'unreliable' and res.stage == 'ood' and 'вне диапазона' in res.reason


# ------------------------------------------------------------------ плохое фото
@pytest.mark.parametrize('kind,light', [('dark', 'темно'), ('bright', 'пересвет'), ('gray', None)])
def test_bad_photo(models, kind, light):
    img = synthetic_eye()[0]
    img = {'dark': img * 0.05, 'bright': np.ones_like(img), 'gray': np.repeat(img.mean(-1, keepdims=True), 3, -1)}[kind]
    res = API.screen(img, 'F', 30, None, models=models)
    _undetermined(res)
    assert res.status == 'retake' and res.stage == 'frame' and res.warnings
    assert res.quality['light'] == light


def test_blurred_tissue(models, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'})
    monkeypatch.setattr(PR, 'quality', lambda *a: ['снимок размыт'])
    res = API.screen(synthetic_eye()[0], 'F', 30, None, models=models)
    _undetermined(res)
    assert res.stage == 'quality' and res.quality['sharpness'] == 'размыто' and 'размыт' in res.reason


@pytest.mark.parametrize('data', [b'', b'not an image', bytearray(b'\xff\xd8garbage'), np.zeros((10, 10, 3), np.uint8)])
def test_unreadable_or_tiny_photo(models, data):
    _undetermined(API.screen(data, None, None, None, models=models))


# ------------------------------------------------------------------ не глаз
def test_not_an_eye(models):
    rng = np.random.default_rng(0)
    skin = np.clip(np.array([0.78, 0.6, 0.5]) + rng.normal(0, 0.03, (300, 400, 3)), 0, 1)
    res = API.screen(skin, 'F', 30, None, models=models)
    _undetermined(res)
    assert res.stage in ('no_eye', 'no_tissue', 'quality')


def test_eye_not_found(models, monkeypatch):
    _route(monkeypatch, set(), eye=None)
    res = API.screen(synthetic_eye()[0], 'F', 30, None, models=models)
    _undetermined(res)
    assert res.stage == 'no_eye' and 'глаз не найден' in res.reason
    assert res.quality['sharpness'] is None and res.quality['tissue_visible'] == {'conjunctiva': False, 'sclera': False}


# ------------------------------------------------------------------ неверный ввод
@pytest.mark.parametrize('age,exc', [('30', TypeError), (True, TypeError), ([30], TypeError), (-1, ValueError),
                                     (200, ValueError), (float('nan'), ValueError), (float('inf'), ValueError)])
def test_wrong_age(models, age, exc):
    with pytest.raises(exc, match='age'):
        API.screen(synthetic_eye()[0], 'F', age, None, models=models)


@pytest.mark.parametrize('age', [30, 30.5, np.int64(30), np.float32(30), None])
def test_numeric_ages_accepted(models, monkeypatch, age):
    _route(monkeypatch, {'conj', 'sclera'})
    assert API.screen(synthetic_eye()[0], 'F', age, None, models=models).status == 'ok'


def test_child_under_5_refused_not_raised(models):
    res = API.screen(synthetic_eye()[0], 'F', 3, None, models=models)
    _undetermined(res)
    assert res.stage == 'age' and 'младше 5 лет' in res.reason


@pytest.mark.parametrize('kw,exc', [
    ({'image': None}, TypeError), ({'image': 'eye.jpg'}, TypeError), ({'image': [1, 2]}, TypeError),
    ({'sex': 'X'}, ValueError), ({'sex': 1}, TypeError),
    ({'pregnant': 'да'}, TypeError), ({'pregnant': True, 'sex': 'M'}, ValueError),
    ({'symptoms': 'слабость'}, TypeError), ({'symptoms': ['']}, TypeError), ({'symptoms': {'a': 'b'}}, TypeError),
    ({'models': None}, TypeError),
])
def test_wrong_input(models, kw, exc):
    args = {'image': synthetic_eye()[0], 'sex': 'F', 'age': 30, 'pregnant': None, 'models': models}
    args.update(kw)
    with pytest.raises(exc):
        API.screen(args.pop('image'), args.pop('sex'), args.pop('age'), args.pop('pregnant'), **args)


# ------------------------------------------------------------------ модель
def test_load_models_rejects_old_bundle(tmp_path):
    path = _save({'models': {'eyes_defy': {'title': 't'}}}, tmp_path)     # без сегментатора конъюнктивы
    with pytest.raises(RuntimeError, match='Я1'):
        API.load_models(path)


def test_load_models_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        API.load_models(tmp_path / 'нет.joblib')
