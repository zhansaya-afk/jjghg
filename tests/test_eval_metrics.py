"""Бинарные метрики evaluate.py (положительный класс — анемия) и внешняя проверка между странами Eyes-defy.
Синтетика: проверяется устройство кода и отсутствие утечки, а не точность."""
import math
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from anemia import core as CORE
from anemia import metrics as M
from anemia import v12_model as V12
from test_core import _synthetic

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import evaluate as EV  # noqa: E402


@pytest.fixture(autouse=True)
def small_models(monkeypatch):
    monkeypatch.setattr(V12, '_clf', lambda: HistGradientBoostingClassifier(max_iter=10, random_state=0))
    monkeypatch.setattr(CORE, '_hb_model', lambda: HistGradientBoostingRegressor(max_iter=10, random_state=0))


def _arrays(tp, fp, fn, tn):
    flag = [True] * tp + [True] * fp + [False] * fn + [False] * tn
    y = [True] * tp + [False] * fp + [True] * fn + [False] * tn
    return np.array(flag), np.array(y)


# ------------------------------------------------------------------ метрики
def test_binary_report_known_counts():
    flag, y = _arrays(8, 2, 2, 88)
    r = M.binary_report(flag, y)
    assert (r['tp'], r['fp'], r['fn'], r['tn']) == (8, 2, 2, 88)
    assert r['acc'] == pytest.approx(0.96) and r['prec'] == pytest.approx(0.8) and r['rec'] == pytest.approx(0.8)
    assert r['spec'] == pytest.approx(88 / 90) and r['npv'] == pytest.approx(88 / 90)
    assert r['f1'] == pytest.approx(0.8) and r['bacc2'] == pytest.approx((0.8 + 88 / 90) / 2)
    for k in ('acc', 'prec', 'rec', 'spec', 'npv'):
        assert r[k + '_lo'] <= r[k] <= r[k + '_hi']
    assert 'f1_lo' not in r                                   # без пациентов — бутстрэпа нет


def test_imbalance_accuracy_is_misleading():
    """95% здоровых: ответ «всегда норма» даёт accuracy 0,95, но не находит ни одного больного."""
    flag, y = _arrays(0, 0, 5, 95)
    r = M.binary_report(flag, y)
    assert r['acc'] == pytest.approx(0.95)
    assert r['rec'] == 0 and r['f1'] == 0 and r['bacc2'] == pytest.approx(0.5)
    assert math.isnan(r['prec']) and math.isnan(r['prec_lo'])   # ни одного ответа «анемия»: PPV не определена


def test_undefined_metrics_are_nan_not_zero():
    flag, y = _arrays(0, 3, 0, 7)                             # больных нет
    r = M.binary_report(flag, y, groups=np.arange(10))
    assert math.isnan(r['rec']) and math.isnan(r['bacc2']) and 'bacc2_lo' not in r
    assert r['f1'] == 0                                       # 3 ложные тревоги: F1 определён и равен 0
    flag, y = _arrays(0, 0, 0, 10)                            # ни больных, ни ответов «анемия»
    r = M.binary_report(flag, y, groups=np.arange(10))
    assert math.isnan(r['f1']) and math.isnan(r['prec']) and r['spec'] == 1
    assert M.binary_report([], []) ['acc'] != M.binary_report([], [])['acc']   # пустая выборка — NaN
    assert EV.fmt(r, 'f1', 2) == '—'                          # в таблице — прочерк, не 0


def test_bootstrap_by_patient():
    rng = np.random.default_rng(0)
    y = rng.random(60) < 0.3
    flag = y ^ (rng.random(60) < 0.15)
    groups = np.repeat(np.arange(30), 2)                      # 2 снимка на пациента
    r = M.binary_report(flag, y, groups=groups)
    assert r['f1_lo'] <= r['f1'] <= r['f1_hi'] and r['bacc2_lo'] <= r['bacc2'] <= r['bacc2_hi']
    assert M.bootstrap_ci(lambda a: float('nan'), groups, 50, 0, y) == (pytest.approx(float('nan'), nan_ok=True),) * 2


def test_table_prints_positive_class(capsys):
    flag, y = _arrays(8, 2, 2, 88)
    EV.print_binary_table([{'task': 'core', 'dataset': 'x', 'method': 'm', 'n': 100, **M.binary_report(flag, y)}])
    out = capsys.readouterr().out
    assert M.POSITIVE_CLASS in out and '8 / 2 / 2 / 88' in out and '| 96 [' in out


# ------------------------------------------------------------------ страны
def _manifest(groups_it=None):
    rows = [{'dataset': 'eyes_defy', 'patient_id': f'ED_India_{i}', 'group': f'ED_India_{i}', 'fold': i % 5}
            for i in range(6)]
    rows += [{'dataset': 'eyes_defy', 'patient_id': f'ED_Italy_{i}', 'group': (groups_it or {}).get(i, f'ED_Italy_{i}'),
              'fold': i % 5} for i in range(4)]
    return pd.DataFrame(rows)


def test_country_split_by_patient():
    A, B, dropped = EV.country_split(_manifest(), 'India', 'Italy')
    assert len(A) == 6 and len(B) == 4 and not set(A.group) & set(B.group) and dropped == []
    assert EV.country_of(A).eq('India').all() and EV.country_of(B).eq('Italy').all()
    A, B, dropped = EV.country_split(_manifest({0: 'ED_India_1'}), 'India', 'Italy')   # копия снимка в обеих странах
    assert dropped == ['ED_India_1'] and 'ED_India_1' not in set(A.group) | set(B.group) and len(A) == 5 and len(B) == 3
    with pytest.raises(AssertionError):
        EV.country_of(pd.DataFrame({'dataset': ['phone'], 'patient_id': ['PH_1']}))


def _two_countries():
    T, _ = _synthetic(80, 0)
    F, S = _synthetic(40, 1)
    F, S = (X.assign(group='q' + X.group.str[1:]) for X in (F, S))
    return T, F, S


def test_external_metrics_rows():
    T, F, S = _two_countries()
    r, O, base = EV.eval_external(T, F, S, 'eyes_defy')
    assert r['n_train'] == 80 and r['n_accepted'] + r['n_ood'] == len(O) == 40
    assert r['tp'] + r['fp'] + r['fn'] + r['tn'] == r['n_accepted']
    assert {'acc', 'prec', 'rec', 'f1', 'bacc2', 'auc', 'sens', 'spec'} <= set(r)
    assert base['n'] == r['n_accepted'] and base['const'] == (T[T.aug == 0].anemic.mean() >= 0.5)
    assert set(O.columns) >= {'p', 'level', 'thr_sens', 'accepted'}


def test_external_no_tuning_on_test():
    """Метки и Hb страны теста не влияют ни на модель, ни на пороги, ни на OOD: перемешали — ответы те же."""
    T, F, S = _two_countries()
    _, O1, _ = EV.eval_external(T, F, S, 'eyes_defy')
    rng = np.random.default_rng(5)
    perm = rng.permutation(F.group.unique())
    lab = F[F.aug == 0].set_index('group')[['anemic', 'hb_g_l', 'y']]
    new = lab.loc[perm].set_index(lab.index)
    F2 = F.drop(columns=['anemic', 'hb_g_l', 'y']).join(new, on='group')
    _, O2, _ = EV.eval_external(T, F2, S, 'eyes_defy')
    for c in ('p', 'level', 'ood', 'thr_sens', 'hb_pred'):
        assert np.allclose(O1[c].values, O2[c].values, equal_nan=True), c


def test_external_empty_route():
    T, F, S = _two_countries()
    assert EV.eval_external(T, pd.DataFrame(), S, 'eyes_defy') == (None, None, None)
    assert EV.eval_external(pd.DataFrame(), F, S, 'eyes_defy') == (None, None, None)


def test_external_rejects_shared_patient():
    T, F, S = _two_countries()
    F = F.assign(group=F.group.where(F.group != 'q0', 'p0'))
    with pytest.raises(AssertionError):
        EV.eval_external(T, F, S, 'eyes_defy')
