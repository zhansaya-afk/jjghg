"""CP-AnemiC исключён (2026-10-19): ребёнок младше 5 лет — отказ, а не модель взрослых; статус беременности,
которого нет в данных, остаётся неизвестным (None) и не превращается в False."""
import math

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from anemia import data as D
from anemia import predict as PR
from anemia.labs import is_anemic, is_pregnant, norm_pregnant, who_limits
from test_core import _bundle, _fake_det
from test_regions import synthetic_eye


# ------------------------------------------------------------------ возраст
@pytest.fixture
def adult_route(monkeypatch):
    """Синтетика: обе ткани «найдены», проверяется только выбор модели по возрасту."""
    monkeypatch.setattr(PR, 'quality', lambda *a: [])
    monkeypatch.setattr(PR, '_detect', lambda rgb, lm=None: _fake_det({'conj', 'sclera'}))
    monkeypatch.setattr(PR, 'learned_conj', lambda det, seg, **kw: det)
    return _bundle()


@pytest.mark.parametrize('age', [0.5, 2, 4.9, 4.999])
def test_under_5_refused(adult_route, age):
    img, _, _ = synthetic_eye()
    r = PR.analyze_photo(img, sex='F', age=age, bundle=adult_route, landmarker=None)
    assert r['status'] == 'unreliable' and r['stage'] == 'age'
    assert r['risk'] is None and r['p_anemia'] is None and r['hb_range'] is None and r['features'] is None
    assert 'младше 5 лет' in r['issues'][0] and 'взрослых' not in r['model']
    assert PR.pick_model(adult_route, age) is None
    assert 'младше 5 лет' in PR.format_result(r)


@pytest.mark.parametrize('age', [5, 5.0, 17, 30, None])
def test_5_and_older_use_adult_model(adult_route, age):
    img, _, _ = synthetic_eye()
    r = PR.analyze_photo(img, sex='F', age=age, bundle=adult_route, landmarker=None)
    assert r['stage'] != 'age'
    assert r['model'] == adult_route['models']['eyes_defy']['title']
    assert PR.pick_model(adult_route, age) == 'eyes_defy'


def test_child_refusal_needs_no_model_file(monkeypatch):
    """Отказ для ребёнка — до загрузки модели: работает, даже если файла модели нет."""
    def no_model(*a, **k):
        raise FileNotFoundError('нет модели')
    monkeypatch.setattr(PR, 'load_bundle', no_model)
    img, _, _ = synthetic_eye()
    assert PR.analyze_photo(img, age=3)['stage'] == 'age'
    with pytest.raises(FileNotFoundError):
        PR.analyze_photo(img, age=30)


# ------------------------------------------------------------------ беременность
@pytest.mark.parametrize('v,want', [(None, None), (float('nan'), None), ('', None), ('?', None), (True, True),
                                    (False, False), (1, True), (0, False), ('да', True), ('нет', False),
                                    ('yes', True), ('False', False), (1.0, True)])
def test_norm_pregnant(v, want):
    assert norm_pregnant(v) is want
    assert is_pregnant(v) is (want is True)


def test_unknown_pregnancy_keeps_non_pregnant_thresholds():
    """Пороги не меняются: неизвестный статус → нормы небеременных; NaN раньше считался беременностью."""
    assert who_limits('F', 30, None).normal == 120
    assert who_limits('F', 30, float('nan')).normal == 120
    assert who_limits('F', 30, True).normal == 110
    assert is_anemic(115, 'F', 30, None) and not is_anemic(115, 'F', 30, True)


def _write_png(path, seed):
    rng = np.random.default_rng(seed)
    Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8)).save(path)


def test_eyes_defy_pregnancy_unknown(tmp_path):
    d = tmp_path / 'eyes_defy_anemia'
    (d / 'raw').mkdir(parents=True)
    rows = []
    for i, (sex, hb) in enumerate([('F', 11.5), ('M', 14.0), ('F', 13.0)]):
        _write_png(d / 'raw' / f'{i}.png', i)
        rows.append({'country': 'Italy', 'pid': str(i), 'image': f'{i}.png', 'mask_palpebral': None,
                     'hb_g_dl': hb, 'sex': sex, 'age': 40})
    pd.DataFrame(rows).to_csv(d / 'labels.csv', index=False)
    assert D.load_eyes_defy(str(tmp_path)).pregnant.map(lambda v: v is None).all()
    man = D.build_manifest(str(tmp_path), n_folds=2)
    assert man.pregnant.map(lambda v: v is None).all()                   # не False
    assert man.anemic.tolist() == [True, False, False]                   # F 115 г/л < 120: нормы небеременных
    man.to_csv(tmp_path / 'manifest.csv', index=False)
    back = pd.read_csv(tmp_path / 'manifest.csv')
    assert back.pregnant.map(lambda v: isinstance(v, float) and math.isnan(v)).all()
    assert not any(is_pregnant(v) for v in back.pregnant)                # NaN из CSV — не беременность


def test_phone_pregnancy_column(tmp_path):
    d = tmp_path / 'phone'
    d.mkdir()
    pd.DataFrame({'file': ['a.png', 'b.png', 'c.png'], 'patient_id': ['1', '2', '3'], 'hb_g_l': [100, 120, 130],
                  'sex': ['F', 'F', 'F'], 'age': [25, 30, 35],
                  'pregnant': ['да', None, 'нет']}).to_csv(d / 'hb.csv', index=False)
    assert D.load_phone(str(tmp_path)).pregnant.tolist() == [True, None, False]
    pd.DataFrame({'file': ['a.png'], 'patient_id': ['1'], 'hb_g_l': [100]}).to_csv(d / 'hb.csv', index=False)
    assert D.load_phone(str(tmp_path)).pregnant.tolist() == [None]       # колонки нет — статус неизвестен


def test_cp_anemic_not_loaded():
    assert set(D.LOADERS) == {'eyes_defy', 'phone'}
