"""Режим «несколько фото» (anemia/series.py): контракт и совпадение с одним фото. Синтетика — не точность."""
import numpy as np
import pytest
from test_api import _route
from test_core import _bundle
from test_regions import synthetic_eye

from anemia import core as CORE
from anemia import predict as PR
from anemia import series as SR


@pytest.fixture(scope='module')
def bundle():
    B = _bundle()
    B['conj_seg'] = 'stub'
    return B


def _img(shift=0.0):
    img = synthetic_eye()[0].astype(np.float32)
    return (np.clip(img + shift, 0, 1) * 255).astype(np.uint8)


def test_scores_match_single_photo(bundle, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'})
    r = PR.analyze_photo(_img(), sex='F', age=30, bundle=bundle, landmarker=None)
    assert r['status'] == 'ok'
    p, _ = SR.photo_scores(r, bundle['models'][SR.model_key(r)])
    assert p == pytest.approx(r['p_anemia'])


def test_series_of_same_photo_equals_single(bundle, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'})
    r = PR.analyze_photo(_img(), sex='F', age=30, bundle=bundle, landmarker=None)
    s = SR.analyze_series([_img(), _img(0.01), _img(0.02)], sex='F', age=30, bundle=bundle, landmarker=None)
    assert s['status'] == 'ok' and s['n_used'] == 3 and s['used'] == [1, 2, 3]
    assert s['risk'] in CORE.RISK_LEVELS and len(s['p_each']) == 3
    assert s['p_anemia'] == pytest.approx(float(np.median(s['p_each'])))
    assert s['model'] == r['model'] and s['tissues'] == r['tissues']
    assert 'медиана' in s['reasons'][0] and isinstance(SR.format_series(s), str)


def test_exact_duplicates_not_counted(bundle, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'})
    s = SR.analyze_series([_img(), _img()], age=30, bundle=bundle, landmarker=None)
    assert s['n_used'] == 1 and s['per_photo'][1]['stage'] == 'duplicate'
    s = SR.analyze_series([_img(), _img()], age=30, bundle=bundle, landmarker=None, min_ok=2)
    assert s['status'] == 'retake' and s['stage'] == 'series' and s['risk'] is None and s['hb_range'] is None


def test_one_photo_equals_single(bundle, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'})
    r = PR.analyze_photo(_img(), sex='F', age=30, bundle=bundle, landmarker=None)
    s = SR.analyze_series([_img()], sex='F', age=30, bundle=bundle, landmarker=None)
    assert s['status'] == 'ok' and s['n_used'] == 1
    assert (s['risk'], s['hb_range']) == (r['risk'], r['hb_range'])
    assert s['p_anemia'] == pytest.approx(r['p_anemia'])
    assert any('одно фото' in n for n in s['notes'])


def _fake(results):
    it = iter(results)
    return lambda ph, **kw: next(it)


def _ok(p_feats, tissues=('conj', 'sclera')):
    return {'status': 'ok', 'stage': 'ok', 'risk': 'средний', 'tissues': list(tissues), 'features': p_feats,
            'issues': [], 'reasons': ['оценено по цвету'], 'notes': [], 'hb_note': 'x', 'model': 'm'}


def _bad(stage='quality'):
    return {'status': 'retake', 'stage': stage, 'risk': None, 'tissues': [], 'issues': ['размыто'], 'notes': []}


def test_rejected_photos_dropped_and_too_few_refused(bundle):
    B = bundle['models']['eyes_defy']
    f = dict(zip(B['feats'], [-6.0, 0.0, 0.0, 0.0]))
    s = SR.analyze_series([_img(i / 100) for i in range(3)], age=30, bundle=bundle, landmarker=None,
                          analyze=_fake([_bad(), _bad(), _bad('no_eye')]))
    assert s['status'] == 'retake' and s['n_used'] == 0 and s['risk'] is None and s['p_anemia'] is None
    assert any('фото № 2' in x for x in s['issues'])
    s = SR.analyze_series([_img(i / 100) for i in range(3)], age=30, bundle=bundle, landmarker=None,
                          analyze=_fake([_ok(f), _bad(), _ok(f)]))
    assert s['status'] == 'ok' and s['used'] == [1, 3] and s['per_photo'][1]['why']


def test_models_not_mixed(bundle):
    B = bundle['models']['eyes_defy']
    f = dict(zip(B['feats'], [-6.0, 0.0, 0.0, 0.0]))
    s = SR.analyze_series([_img(i / 100) for i in range(3)], age=30, bundle=bundle, landmarker=None,
                          analyze=_fake([_ok(f), _ok({'c_EI_m': -6.0, 'male': 0.0}, ('conj',)), _ok(f)]))
    assert s['used'] == [1, 3] and 'другой моделью' in s['per_photo'][1]['why']


def test_child_refused_once(bundle):
    s = SR.analyze_series([_img(), _img(0.01)], age=2, bundle=bundle, landmarker=None)
    assert s['stage'] == 'age' and s['risk'] is None


def test_pick_model_and_empty():
    assert SR.pick_model(['eyes_defy_conj', 'eyes_defy', 'eyes_defy_conj']) == 'eyes_defy_conj'
    assert SR.pick_model(['eyes_defy_conj', 'eyes_defy']) == 'eyes_defy'
    assert SR.analyze_series([], bundle={})['status'] == 'retake'


def test_notebook_cell_compiles_and_uses_series():
    import json
    import os
    with open(os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'colab_train.ipynb'), encoding='utf-8') as f:
        nb = json.load(f)
    titles = [''.join(c['source']) for c in nb['cells']]
    i = next(k for k, s in enumerate(titles) if s.startswith('#@title Ячейка Я2с '))
    assert titles[i - 1].startswith('#@title Ячейка Я2 ')          # сразу после ячейки одного фото
    compile(titles[i], 'Я2с', 'exec')
    assert 'analyze_series(' in titles[i] and 'method=метод)' in titles[i]
