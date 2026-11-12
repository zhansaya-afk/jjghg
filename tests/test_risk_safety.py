"""Риск без ложной уверенности (2026-11-12): пол не указан, NaN / OOD / нет зоны «низкий». Синтетика проверяет
маршрут и правила, не точность."""
import numpy as np
import pytest
from test_core import _fake_det, _synthetic
from test_regions import synthetic_eye

from anemia import core as CORE
from anemia import predict as PR


@pytest.fixture(scope='module')
def bundle():
    F, _ = _synthetic()
    F = F.rename(columns={'c_x': 'c_EI_m', 'c_z': 'c_a_m', 's_x': 's_EI_m'})
    m = {'eyes_defy': CORE.fit_core(F, ['c_EI_m', 'c_a_m', 's_EI_m', 'male'], 'eyes_defy'),
         'eyes_defy_conj': CORE.fit_core(CORE.usable(F, 'eyes_defy_conj'), ['c_EI_m', 'c_a_m', 'male'], 'eyes_defy_conj')}
    return {'models': m, 'conj_seg': 'stub'}


@pytest.fixture
def route(monkeypatch):
    monkeypatch.setattr(PR, '_detect', lambda rgb, lm=None: _fake_det({'conj', 'sclera'}))
    monkeypatch.setattr(PR, 'learned_conj', lambda det, seg, **k: det)
    monkeypatch.setattr(PR, 'quality', lambda *a: [])
    monkeypatch.setattr(CORE, 'ood_share', lambda x, lo, hi: np.zeros(len(x)))


def _run(bundle, sex):
    return PR.analyze_photo(synthetic_eye()[0], sex=sex, age=30, bundle=bundle, landmarker=None)


def test_unknown_sex_is_scored_as_female_and_user_is_told(bundle, route):
    """Пол не указан → признак male 0,5, деревья HGB относят его к женщинам (P та же, что у 'F'); норма ВОЗ — женская.
    Пользователь должен это видеть: у мужчины без пола риск завышен."""
    r_none, r_f = _run(bundle, None), _run(bundle, 'F')
    assert r_none['status'] == r_f['status'] == 'ok'
    assert r_none['p_anemia'] == pytest.approx(r_f['p_anemia'])
    assert any('пол не указан' in n and 'женщин' in n for n in r_none['notes'])
    assert not any('пол не указан' in n for n in r_f['notes'] + _run(bundle, 'M')['notes'])


def test_nan_probability_is_never_low():
    """Регрессионный замок: P не число → «средний», не «низкий»."""
    assert CORE.risk_level([np.nan], 0.3, 0.8)[0] == 1


def test_no_low_zone_means_never_low():
    """Регрессионный замок: модель без зоны «низкий» (t_lo = −inf) не выдаёт «низкий» ни при какой P."""
    assert (CORE.risk_level(np.linspace(0, 1, 11), -np.inf, 0.9) >= 1).all()


def test_ood_gives_no_risk_and_no_hb(bundle, route, monkeypatch):
    """Регрессионный замок: снимок не похож на обучающие → ни уровня риска, ни P, ни Hb."""
    monkeypatch.setattr(CORE, 'ood_share', lambda x, lo, hi: np.ones(len(x)))
    r = _run(bundle, 'F')
    assert r['status'] == 'unreliable' and r['stage'] == 'ood'
    assert r['risk'] is None and r['p_anemia'] is None and r['hb_range'] is None
