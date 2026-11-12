"""Эксперимент 2026-11-12 (scripts/experiments/country_relative.py): страновое перевзвешивание и относительные
признаки. Синтетика проверяет только код (равенство baseline ядру, баланс стран, отсутствие утечки), не точность."""
import os
import sys

import numpy as np
import pandas as pd
import pytest
from test_core import _synthetic

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts', 'experiments'))
import country_relative as X

from anemia import core as CORE


def _with_country(F):
    F = F.copy()
    i = F.group.str[1:].astype(int)
    F['group'] = np.where(i % 3 == 0, 'ED_Italy_', 'ED_India_') + i.astype(str)
    return F


def test_baseline_fit_equals_production_fit_core():
    F, _ = _synthetic()
    feats = ['c_x', 'c_z', 's_x', 'male']
    A, B = X.fit_variant(F, feats, 'eyes_defy', balance=False), CORE.fit_core(F, feats, 'eyes_defy')
    for k in ('t_lo', 't_hi', 'thr_sens', 'hb_mae_cv', 'hb_halfwidth', 'hb_ok'):
        assert A[k] == pytest.approx(B[k]), k
    assert A['cal'] == pytest.approx(B['cal'])
    X0 = F[F.aug == 0][feats].values
    for a, b in zip(CORE.apply_core(A, X0)[:3], CORE.apply_core(B, X0)[:3]):
        np.testing.assert_allclose(a, b)


def test_country_balance_makes_label_independent_of_country():
    F = _with_country(_synthetic()[0])
    T = X.country_balance(F, seed=0)
    c = X.country(T.group)
    share = T.groupby(c).anemic.mean()
    assert share.round(6).nunique() == 1 and share.iloc[0] == pytest.approx(0.5)
    assert set(T.group) == set(F.group)                               # каждый пациент остался, новых нет
    assert (T.groupby([c, 'anemic']).size().nunique()) == 1          # все ячейки страна × класс равны


def test_country_balance_needs_both_classes_in_each_country():
    F = _with_country(_synthetic()[0])
    F = F[~(X.country(F.group).eq('Italy') & F.anemic)]
    with pytest.raises(ValueError, match='Italy'):
        X.country_balance(F, seed=0)


def test_relative_features_are_conj_minus_sclera_and_nan_without_sclera():
    F = pd.DataFrame({'group': ['a', 'b'], 'c_EI_m': [0.3, 0.2], 's_EI_m': [0.1, np.nan], 'c_L_p50': [50.0, 40.0],
                      's_L_p50': [80.0, np.nan], 'c_only_m': [1.0, 1.0], 'w_EI_m': [0.25, np.nan], 'male': [1.0, 0.0]})
    R = X.add_relative(F)
    assert R.d_EI_m.tolist()[0] == pytest.approx(0.2) and np.isnan(R.d_EI_m.iloc[1])
    assert R.d_L_p50.iloc[0] == pytest.approx(-30.0)
    assert 'd_only_m' not in R                                         # у склеры нет пары — разности нет
    f = X.relative_feature_list(R.columns)
    assert set(f) == {'d_EI_m', 'd_L_p50', 'w_EI_m', 'male'}
    assert X.relative_ok(R).tolist() == [True, False]


def test_cv_refuses_test_patient_in_train():
    F, _ = _synthetic()
    with pytest.raises(AssertionError, match='утечка'):
        X.cv(F, ['c_x', 'male'], 'eyes_defy', {f: F for f in range(5)}, balance=False)


def test_cv_oof_covers_each_patient_once_on_manifest_folds():
    F, _ = _synthetic()
    train = {f: F[F.fold != f] for f in range(5)}
    O = X.cv(F, ['c_x', 'c_z', 'male'], 'eyes_defy', train, balance=False)
    assert len(O) == 80 and O.group.is_unique and set(O.level.unique()) <= {0, 1, 2}
    ref, _ = CORE.cross_validate(F, ['c_x', 'c_z', 'male'], 'eyes_defy', train_by_fold=train)
    np.testing.assert_allclose(O.sort_values('group').p.values, ref.sort_values('group').p.values)


def test_paired_delta_zero_for_identical_and_signed():
    g = np.array([f'p{i}' for i in range(40)])
    y = np.arange(40) % 2 == 0
    a = np.where(y, 1, 0)
    d, lo, hi, dropped = X.paired_delta(lambda lv, yy: X.spec(lv, yy), g, a, a, y, n_boot=200)
    assert d == lo == hi == 0 and dropped == 0
    b = a.copy()
    b[~y] = 1                                                         # все здоровые → «средний+»: спец. падает до 0
    d, lo, hi, dropped = X.paired_delta(lambda lv, yy: X.spec(lv, yy), g, a, b, y, n_boot=200)
    assert d == pytest.approx(-1.0) and hi < 0 and dropped == 0


def _jobs():
    """Синтетические «задания» evaluate.core_jobs: test-строки + обучающие строки для каждого внешнего фолда."""
    F, S = _synthetic(n=60)
    F, S = _with_country(F), _with_country(S)
    for D in (F, S):
        D['c_a_m'], D['s_a_m'], D['w_a_m'] = D.c_x, D.s_x, D.c_x - D.s_x.fillna(0)
        D['who_normal'] = 120.0
    J = []
    for g, d in F.groupby('group'):
        f = int(d.fold.iloc[0])
        key = 'eyes_defy' if d.s_x.notna().all() else 'eyes_defy_conj'
        sr = S[S.group == g].to_dict('records')
        J.append({'i': g, 'role': 'test', 'key': key, 'why': '', 'rows': d.to_dict('records'), 'srows': sr})
        for o in range(5):
            if o != f:
                J.append({'i': g, 'role': o, 'key': key, 'why': '', 'rows': d.to_dict('records'), 'srows': []})
    df = F[F.aug == 0][['group', 'fold']].assign(patient_id=lambda d: d.group, sex='F', dataset='eyes_defy')

    class EV:
        @staticmethod
        def _frame(J, role, key=None, s=False):
            sel = [x for x in J if x['role'] == role and x['key'] is not None and (key is None or x['key'] == key)]
            return pd.DataFrame([r for x in sel for r in (x['srows'] if s else x['rows'])])
    return EV, df.reset_index(drop=True), J


def test_main_runs_end_to_end_on_synthetic_jobs(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(X, 'build', lambda root, jobs: _jobs())
    monkeypatch.setattr(sys, 'argv', ['x', '--out', str(tmp_path), '--seeds', '1', '--n-boot', '30'])
    X.main()
    R = pd.read_csv(tmp_path / 'summary_by_seed.csv')
    assert set(R['вариант']) == set(X.VARIANTS) and len(R) == 4 and (R.n == R.n.iloc[0]).all()   # одни пациенты
    D = pd.read_csv(tmp_path / 'paired_delta_vs_baseline.csv')
    assert set(D['часть']) == {'все', 'India', 'Italy'}
    cov = pd.read_json(tmp_path / 'coverage.json', typ='series')
    assert cov['только конъюнктива (склеры нет → относительных признаков нет)'] > 0
    assert 'страна × пол: India F' in set(pd.read_csv(tmp_path / 'subgroups_C_относит.csv')['группа'])


def _delta(v, part, metric, d, lo, hi, n_healthy=40, dropped=0):
    return {'вариант': v, 'часть': part, 'метрика': metric, 'Δ': d, 'lo': lo, 'hi': hi, 'n_здоровых': n_healthy,
            'отброшено_повторов': dropped}


def test_verdict_rule_prespecified():
    seeds = pd.DataFrame({'вариант': ['A_baseline'] * 3 + ['B'] * 3, 'FP': [29, 30, 28, 20, 21, 19]})
    good = pd.DataFrame([_delta('B', 'India', 'спец', .3, .1, .5), _delta('B', 'все', 'чувств', -.02, -.08, .03),
                         _delta('B', 'все', 'AUC', 0, -.02, .02)])
    assert X.verdict(good, seeds)['B'].startswith('принять')
    fn_cost = good.copy()
    fn_cost.loc[1, ['Δ', 'lo']] = [-.08, -.18]                        # FP снизился ценой FN → не принимать
    assert 'FN' in X.verdict(fn_cost, seeds)['B']
    weak = good.copy()
    weak.loc[0, 'lo'] = -.05                                          # ДИ Индии задевает 0 → не доказано
    assert X.verdict(weak, seeds)['B'].startswith('не доказано')


# ------------------------------------------------------------------ правки 2026-11-12 (блокеры проверки готовности)
def test_build_passes_per_fold_anatomy_limits_like_evaluate_core(monkeypatch, tmp_path):
    """Пределы анатомии — по маскам врачей обучающих фолдов (EV.anatomy_limits по ключам сегментаторов), и именно они
    уходят в core_jobs: так же, как evaluate --task core. Глобальные AN.LIMITS (по всем 217 маскам) не используются."""
    import types
    calls = {}
    segs = {(0,): 's0', (1,): 's1', (0, 1): 's01'}
    lims = {k: {'wfrac_min': 0.4, 'tag': k} for k in segs}
    ev = types.SimpleNamespace(
        FeatureCache=lambda *a, **k: 'cache', source_version=lambda p: 'v', __file__=str(tmp_path / 'evaluate.py'),
        conj_segmenters=lambda root, df, jobs, cache, timer, nested, full, external: (
            calls.setdefault('seg', {'nested': nested, 'full': full, 'external': external}), segs)[1],
        anatomy_limits=lambda root, df, jobs, keys, cache, external=True: (
            calls.setdefault('lim', {'keys': list(keys), 'external': external}), lims)[1],
        core_jobs=lambda root, df, jobs, s, cache, timer, nested, limits=None: (
            calls.setdefault('jobs', {'nested': nested, 'limits': limits, 'segs': s}), [])[1])
    monkeypatch.setitem(sys.modules, 'evaluate', ev)
    pd.DataFrame({'dataset': ['eyes_defy'], 'patient_id': ['ED_India_1']}).to_csv(tmp_path / 'manifest.csv', index=False)
    X.build(str(tmp_path), 1)
    assert calls['seg'] == {'nested': True, 'full': False, 'external': True}
    assert calls['lim'] == {'keys': list(segs), 'external': True}
    assert calls['jobs']['nested'] is True and calls['jobs']['limits'] is lims and calls['jobs']['segs'] is segs


def _runs():
    """3 пациента India (H1, H2 здоровы, A1 болен) + I1 Италия; OOD: C seed 1 — H2, B seed 0 — I1."""
    g = ['ED_India_H1', 'ED_India_H2', 'ED_India_A1', 'ED_Italy_I1']
    def O(rej=()):
        return pd.DataFrame({'group': g, 'accepted': [x not in rej for x in g]})
    return {('A', 0): (O(), 5), ('A', 1): (O(), 5), ('B', 0): (O({'ED_Italy_I1'}), 5), ('B', 1): (O(), 5),
            ('C', 0): (O(), 3), ('C', 1): (O({'ED_India_H2'}), 3)}


def test_coverage_counts_exclusions_per_variant_seed_and_reasons():
    J = [{'role': 'test', 'key': 'eyes_defy', 'why': ''}] * 4 + [
        {'role': 'test', 'key': 'eyes_defy_conj', 'why': ''},
        {'role': 'test', 'key': None, 'why': 'качество: снимок размыт'},
        {'role': 'test', 'key': None, 'why': 'качество: слишком темно'},
        {'role': 'test', 'key': None, 'why': 'нет тканей: конъюнктива не найдена сегментатором'},
        {'role': 0, 'key': None, 'why': 'не тест — не считать'}]
    runs = _runs()
    common = set.intersection(*(set(o.group[o.accepted]) for o, _ in runs.values()))
    c = X.coverage(8, J, runs, common)
    assert c['пациентов в манифесте'] == 8
    assert c['причины отказа маршрута'] == {'качество': 2, 'нет тканей': 1}
    assert c['только конъюнктива (склеры нет → относительных признаков нет)'] == 1
    v = c['по вариантам и seed']
    assert v['B seed 0']['OOD'] == 1 and v['B seed 0']['исключено всего'] == 5             # 4 вне маршрута + 1 OOD
    assert v['B seed 0']['доля исключённых'] == pytest.approx(5 / 8)
    assert v['A seed 0']['исключено всего'] == 4 and v['A seed 0']['OOD'] == 0
    assert c['общий набор'] == {'n': 2, 'доля от манифеста': pytest.approx(2 / 8)}
    out = c['исключено из общего набора из-за OOD других вариантов']
    assert out['A'] == 2 and out['C'] == 1 and out['B'] == 1        # A сам принял всех 4, в общем наборе только 2


def test_paired_delta_reports_dropped_replicates():
    """Бутстрэп-повторы без здоровых (спец. не определена) считаются и возвращаются, а не теряются молча."""
    g = np.array(['h', 'a1', 'a2', 'a3', 'a4'])
    y = np.array([False, True, True, True, True])
    lv = np.array([0, 1, 1, 1, 1])
    _, _, _, dropped = X.paired_delta(X.spec, g, lv, lv, y, n_boot=200)
    assert dropped > 0                                                # повторы без единственного здорового бывают


def test_verdict_small_india_healthy_never_above_not_proven():
    seeds = pd.DataFrame({'вариант': ['A_baseline'] * 3 + ['B'] * 3, 'FP': [29, 30, 28, 20, 21, 19]})
    good = pd.DataFrame([_delta('B', 'India', 'спец', .3, .1, .5, n_healthy=26),
                         _delta('B', 'все', 'чувств', -.02, -.08, .03), _delta('B', 'все', 'AUC', 0, -.02, .02)])
    v = X.verdict(good, seeds)['B']
    assert v.startswith('не доказано') and 'n=26' in v and '30' in v


def test_verdict_dropped_replicates_block_accept():
    seeds = pd.DataFrame({'вариант': ['A_baseline'] * 3 + ['B'] * 3, 'FP': [29, 30, 28, 20, 21, 19]})
    good = pd.DataFrame([_delta('B', 'India', 'спец', .3, .1, .5, n_healthy=40, dropped=3),
                         _delta('B', 'все', 'чувств', -.02, -.08, .03), _delta('B', 'все', 'AUC', 0, -.02, .02)])
    v = X.verdict(good, seeds)['B']
    assert v.startswith('не доказано') and 'отброшено 3' in v


def test_verdict_fn_rejection_kept_for_small_india():
    """Ограничение n < 30 не повышает статус: отказ из-за роста FN остаётся отказом."""
    seeds = pd.DataFrame({'вариант': ['A_baseline'] * 3 + ['B'] * 3, 'FP': [29, 30, 28, 20, 21, 19]})
    bad = pd.DataFrame([_delta('B', 'India', 'спец', .3, .1, .5, n_healthy=26),
                        _delta('B', 'все', 'чувств', -.08, -.18, .0), _delta('B', 'все', 'AUC', 0, -.02, .02)])
    assert X.verdict(bad, seeds)['B'].startswith('отклонить')
