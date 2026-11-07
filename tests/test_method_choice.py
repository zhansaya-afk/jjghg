"""Выбор метода анализа (список «Ткань» в Я2 → analyze_photo(method=…) / api.screen(method=…)): маршрут и отказы
на синтетике (проверяется маршрут, не точность)."""
import re

import numpy as np
import pytest
from test_core import _fake_det, _synthetic
from test_regions import synthetic_eye

from anemia import api as API
from anemia import core as CORE
from anemia import predict as PR


@pytest.fixture(scope='module')
def bundle():
    F, _ = _synthetic()
    F = F.rename(columns={'c_x': 'c_EI_m', 'c_z': 'c_a_m', 's_x': 's_EI_m'})
    m = {'eyes_defy': CORE.fit_core(F, ['c_EI_m', 'c_a_m', 's_EI_m', 'male'], 'eyes_defy'),
         'eyes_defy_conj': CORE.fit_core(CORE.usable(F, 'eyes_defy_conj'), ['c_EI_m', 'c_a_m', 'male'], 'eyes_defy_conj'),
         'eyes_defy_sclera': CORE.fit_core(CORE.usable(F, 'eyes_defy_sclera'), ['s_EI_m', 'male'], 'eyes_defy_sclera')}
    return {'models': m, 'conj_seg': 'stub'}


def _route(monkeypatch, use, bad=()):
    """Найдены ткани use; ткани из bad — «плохое качество». OOD выключен (малая модель учена не на этих признаках)."""
    monkeypatch.setattr(PR, '_detect', lambda rgb, lm=None: _fake_det(use))
    monkeypatch.setattr(PR, 'learned_conj', lambda det, seg: det)
    monkeypatch.setattr(CORE, 'ood_share', lambda x, lo, hi: np.zeros(len(x)))

    def quality(crop, a, b, side):
        return ['размыто'] if bad and b is not None and not b.any() and any(
            (a == _fake_det({t})[t]).all() and _fake_det({t})[t].any() for t in bad) else []
    monkeypatch.setattr(PR, 'quality', quality)


def _run(bundle, method):
    return PR.analyze_photo(synthetic_eye()[0], sex='F', age=30, bundle=bundle, landmarker=None, method=method)


@pytest.mark.parametrize('method,key,tissues', [('conj', 'eyes_defy_conj', ['conj']),
                                                ('sclera', 'eyes_defy_sclera', ['sclera']),
                                                ('both', 'eyes_defy', ['conj', 'sclera'])])
def test_method_selects_its_model_and_tissues(bundle, monkeypatch, method, key, tissues):
    _route(monkeypatch, {'conj', 'sclera'})
    r = _run(bundle, method)
    assert r['status'] == 'ok', (r['stage'], r['issues'])
    assert r['model'] == bundle['models'][key]['title'] and r['tissues'] == tissues and r['method'] == method
    assert set(r['features']) == set(bundle['models'][key]['feats'])
    assert any(PR.METHOD_RU[method] in x for x in r['reasons'])
    assert f'Метод анализа: {PR.METHOD_RU[method]}' in PR.format_result(r)


def test_auto_unchanged(bundle, monkeypatch):
    """method=None (и 'auto') — прежнее поведение: обе пригодные ткани → модель обеих."""
    _route(monkeypatch, {'conj', 'sclera'})
    for m in (None, 'auto'):
        r = _run(bundle, m)
        assert r['tissues'] == ['conj', 'sclera'] and r['method'] == 'auto'
        assert 'Метод анализа' not in PR.format_result(r)


@pytest.mark.parametrize('method,use,missing', [('sclera', {'conj'}, 'склера'), ('both', {'conj'}, 'склера'),
                                                ('conj', {'sclera'}, 'конъюнктива'), ('both', {'sclera'}, 'конъюнктива')])
def test_missing_tissue_refused_not_substituted(bundle, monkeypatch, method, use, missing):
    _route(monkeypatch, use)
    r = _run(bundle, method)
    assert r['status'] != 'ok' and r['risk'] is None and r['hb_range'] is None and r['p_anemia'] is None
    assert r['stage'] == 'no_tissue' and r['tissues'] == []
    assert any(PR.METHOD_RU[method] in x and missing in x and 'не подставляется' in x for x in r['issues'])


def test_poor_quality_tissue_refused(bundle, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'}, bad=('sclera',))
    r = _run(bundle, 'sclera')
    assert r['stage'] == 'quality' and r['risk'] is None
    assert any('склера: плохое качество' in x for x in r['issues'])
    assert _run(bundle, 'conj')['status'] == 'ok'                  # конъюнктива пригодна — её метод работает


def test_wrong_method_raises(bundle):
    with pytest.raises(ValueError):
        _run(bundle, 'iris')


def test_api_passes_method(bundle, monkeypatch):
    seen = {}
    real = PR.analyze_photo

    def spy(*a, **k):
        seen['method'] = k.get('method')
        return real(*a, **k)
    monkeypatch.setattr(PR, 'analyze_photo', spy)
    _route(monkeypatch, {'conj', 'sclera'})
    models = API.Models(bundle=bundle, landmarker=None, version='core-test')
    res = API.screen(synthetic_eye()[0], 'F', 30, None, models=models, method='sclera')
    assert seen['method'] == 'sclera' and res.tissue == 'склера'
    API.screen(synthetic_eye()[0], 'F', 30, None, models=models)     # без method — как раньше
    assert seen['method'] is None
    with pytest.raises(ValueError):
        API.screen(synthetic_eye()[0], 'F', 30, None, models=models, method='x')


@pytest.mark.parametrize('use,key', [({'conj', 'sclera'}, 'eyes_defy'), ({'conj'}, 'eyes_defy_conj'),
                                     ({'sclera'}, 'eyes_defy_sclera')])
def test_auto_picks_model_by_available_tissues(bundle, monkeypatch, use, key):
    """«Автоматический выбор» = прежняя логика: модель по пригодным тканям, в т.ч. когда одной ткани нет."""
    _route(monkeypatch, use)
    r = _run(bundle, 'auto')
    assert r['status'] == 'ok' and r['model'] == bundle['models'][key]['title'] and r['method'] == 'auto'
    r0 = _run(bundle, None)                                          # 'auto' = вызов без method (старый API)
    assert (r0['model'], r0['tissues'], r0['p_anemia'], r0['risk']) == (r['model'], r['tissues'], r['p_anemia'], r['risk'])


def test_auto_falls_back_to_conj_when_sclera_poor(bundle, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'}, bad=('sclera',))
    r = _run(bundle, 'auto')
    assert r['status'] == 'ok' and r['tissues'] == ['conj']        # прежнее поведение авто: непригодная ткань не нужна


# ------------------------------------------------------------------ интерфейс: выпадающий список «Ткань» в Я2
def _cell(title):
    import json
    import os
    with open(os.path.join(os.path.dirname(__file__), '..', 'notebooks', 'colab_train.ipynb'), encoding='utf-8') as f:
        nb = json.load(f)
    return next(''.join(c['source']) for c in nb['cells'] if ''.join(c['source']).startswith(title))


def test_notebook_ya2_tissue_param_default_auto():
    s = _cell('#@title Ячейка Я2 ')
    line = next(x for x in s.splitlines() if x.startswith('Ткань ='))
    assert line.startswith("Ткань = 'Авто'") and '["Авто", "Конъюнктива", "Склера", "Обе"]' in line
    m = re.search(r"метод = (\{.*?\})\[Ткань\]", s)
    mapping = eval(m.group(1))
    assert list(mapping) == ['Авто', 'Конъюнктива', 'Склера', 'Обе']
    assert mapping['Авто'] == 'auto' and PR.check_method('auto') is None
    assert {v for k, v in mapping.items() if k != 'Авто'} == set(PR.METHODS)
    assert 'method=метод)' in s
