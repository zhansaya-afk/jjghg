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


# ---- метрики масок по снимкам (seg_per_image.csv / seg_summary.csv) ----------------------------------------------
from anemia import anatomy as AN  # noqa: E402
from anemia import predict as PR  # noqa: E402
from anemia import seg_metrics as SM  # noqa: E402
from test_anatomy import blob, strip  # noqa: E402


def _old_seg_scores(model, items, limits=None):
    """Копия _seg_scores до 2026-11-03: строки истории seg_ext должны считаться так же."""
    from anemia import segment_baseline as SB
    out = []
    for rgb, gt in items:
        if not gt.any():
            out.append(None)
            continue
        p = SB.conj_mask(model, rgb)
        if limits is not None and p.any():
            an = AN.check(p, None, limits)
            p = an['conj'] if an['conj_ok'] else np.zeros_like(p)
        out.append((M.iou(p, gt), M.dice(p, gt), bool(p.any())))
    return out


def _raw_by_tag(monkeypatch, raws):
    """SB.conj_mask(model, rgb) → маска по метке снимка rgb (вместо обученного сегментатора)."""
    monkeypatch.setattr(EV.SB, 'conj_mask', lambda model, rgb: raws[rgb].copy())


def test_seg_one_statuses(monkeypatch):
    gt = strip()
    lid = strip(x0=40, x1=200)
    _raw_by_tag(monkeypatch, {'lid': lid, 'empty': np.zeros_like(gt), 'spot': blob(120, 150, 40)})
    st, p, acc, why, old = EV._seg_one(None, 'lid', gt, None)
    assert (st, acc, why) == ('ok', True, '') and old == (M.iou(lid, gt), M.dice(lid, gt), True)
    st, p, acc, why, old = EV._seg_one(None, 'empty', gt, AN.LIMITS)
    assert (st, acc, why, old) == ('ok', False, 'сегментатор не нашёл веко', (0.0, 0.0, False)) and not p.any()
    st, p, acc, why, old = EV._seg_one(None, 'spot', gt, AN.LIMITS)       # узкое «пятно» — отказ анатомии
    assert st == 'refused' and not acc and why and p.any() and old == (0.0, 0.0, False)
    assert EV._seg_one(None, 'spot', gt, None)[0] == 'ok'                # без анатомии то же пятно принимается
    assert EV._seg_one(None, 'lid', np.zeros_like(gt), AN.LIMITS) is None
    r = EV._seg_record('m', 'eyes_defy', 'ED_India_1', 'g', 0, np.zeros_like(gt), None)
    assert r['status'] == 'no_gt' and r['why'] == 'эталонная маска пуста'
    r = EV._seg_record('m', 'eyes_defy', 'ED_India_1', 'g', 0, gt, EV._seg_one(None, 'spot', gt, AN.LIMITS))
    assert r['status'] == 'refused' and r['country'] == 'India' and r['pred_empty'] is False


def test_seg_scores_history_rule_unchanged(monkeypatch):
    gt = strip()
    _raw_by_tag(monkeypatch, {'lid': strip(x0=40, x1=200), 'empty': np.zeros_like(gt), 'spot': blob(120, 150, 40)})
    items = [('lid', gt), ('empty', gt), ('spot', gt), ('lid', np.zeros_like(gt))]
    for lim in (None, AN.LIMITS):
        assert EV._seg_scores(None, items, lim) == _old_seg_scores(None, items, lim)


class _TrainedOn:
    def __init__(self, train):
        self.train = set(train)


def test_seg_ext_records_no_fold_leak(monkeypatch):
    ids = [f'ED_India_{k}' for k in range(5)] + [f'ED_Italy_{k}' for k in range(5)]
    df = pd.DataFrame({'patient_id': ids, 'group': ids, 'fold': [k % 5 for k in range(10)],
                       'image': [f'{i}.jpg' for i in ids], 'mask_conj': [f'{i}_m.png' for i in ids]})
    idx = pd.DataFrame({'id': [f'phone_conj_masks_{k}' for k in range(6)], 'group': [f'p{k // 2}' for k in range(6)],
                        'file': [f'Ivan Petrov {k}.jpg' for k in range(6)], 'mask': [f'{k}.png' for k in range(6)]})
    gt = strip()
    raws = {('ed', i): strip(x0=40, x1=200) for i in ids} | {('ex', i): strip(x0=40, x1=200) for i in idx['id']}
    raws[('ed', 'ED_India_1')] = np.zeros_like(gt)                     # сегментатор ничего не нашёл
    raws[('ex', 'phone_conj_masks_3')] = blob(120, 150, 40)           # пятно: отказ только с анатомией
    empty_gt = {('ed', 'ED_Italy_0'), ('ex', 'phone_conj_masks_5')}
    monkeypatch.setattr(EV, 'BAD_DOCTOR_MASKS', {'ED_Italy_4': 'испорчена'})
    monkeypatch.setattr(EV.EXT, 'seg_index', lambda root: idx)
    monkeypatch.setattr(EV, '_segpix_job', lambda root, r, i, cache: (('ed', r['patient_id']), {}))
    monkeypatch.setattr(EV, 'external_seg_samples', lambda root, jobs, cache, idx: [('ex', i) for i in idx['id']])

    def item(root, r, kind):
        tag = ('ed', r['patient_id']) if kind == 'ed' else ('ex', r['id'])
        return tag, (np.zeros_like(gt) if tag in empty_gt else gt)

    def conj_mask(model, tag):
        assert tag not in model.train, f'утечка: {tag} в обучении своего фолда'
        return raws[tag].copy()

    monkeypatch.setattr(EV, '_seg_test_item', item)
    monkeypatch.setattr(EV.SB, 'fit_samples', _TrainedOn)
    monkeypatch.setattr(EV.SB, 'conj_mask', conj_mask)
    monkeypatch.setattr(EV.AN, 'fit_limits', lambda stats: AN.LIMITS)
    rows, recs, err = EV.run_seg_external('root', df, 1)
    assert err is None
    P = SM.records_frame(recs)
    assert P.arm.nunique() == 3
    assert not P.arm.str.contains('отказ = пустая маска').any()      # в новых CSV отказ отделён от пустой маски
    assert sum('(отказ = пустая маска)' in r['method'] for r in rows) == 2   # имя в истории прежнее
    assert not P['id'].str.contains('Ivan|jpg|png').any()           # только id индекса, не имя файла
    for arm, d in P.groupby('arm'):
        ed, ph = d[d.dataset == 'eyes_defy'].set_index('id'), d[d.dataset == 'phone_conj_masks'].set_index('id')
        assert len(ed) == 10 and len(ph) == 6 and ed.index.is_unique and ph.index.is_unique
        assert ed.loc['ED_Italy_4', 'status'] == 'no_gt' and 'BAD_DOCTOR_MASKS' in ed.loc['ED_Italy_4', 'why']
        assert ed.loc['ED_Italy_0', 'status'] == 'no_gt' and ph.loc['phone_conj_masks_5', 'status'] == 'no_gt'
        assert ed.loc['ED_India_1', 'status'] == 'ok' and ed.loc['ED_India_1', 'pred_empty']
        assert ed.loc['ED_India_1', 'iou'] == 0 and np.isnan(ed.loc['ED_India_1', 'precision'])
        assert (ed.fold.to_dict() == df.set_index('patient_id').fold.to_dict())
        assert ph.loc['phone_conj_masks_3', 'status'] == ('refused' if 'анатомия' in arm else 'ok')
        assert ph.loc['phone_conj_masks_3', 'stage'] == ('anatomy' if 'анатомия' in arm else 'accepted')
        assert ed.loc['ED_India_1', 'stage'] == 'empty_mask'
        hist = arm.replace(' + анатомия', ' + анатомия (отказ = пустая маска)')
        row = next(r for r in rows if r['method'] == hist and r['dataset'] == 'phone_conj_masks')
        valid = ph[ph.status != 'no_gt']
        old_iou = np.where(valid.status == 'refused', 0.0, valid.iou)   # история: отказ = IoU 0, как раньше
        assert row['n'] == len(valid) and row['iou_mean'] == pytest.approx(old_iou.mean())


class _FoldSeg:
    def __init__(self, key, trap=False):
        self.key, self.trap = key, trap


class _DetCache:
    def __init__(self, dets):
        self.dets = dets

    def fetch(self, kind, files, arrays, params, func, *args):
        return self.dets[files['image']], {}


def _core_case(monkeypatch):
    """6 снимков, 3 фолда: принят / отказ до сегментатора / пустая маска / модель без конъюнктивы / BAD / без маски."""
    h = w = 20
    gt = np.zeros((h, w), bool)
    gt[8:14, 2:18] = True
    conj = np.zeros((h, w), bool)
    conj[9:15, 2:18] = True
    ids = ['ED_India_0', 'ED_India_1', 'ED_Italy_2', 'ED_Italy_3', 'ED_India_4', 'ED_Italy_5']
    df = pd.DataFrame({'dataset': 'eyes_defy', 'patient_id': ids, 'group': ids, 'fold': [0, 0, 1, 1, 2, 2],
                       'image': [f'i{k}.jpg' for k in range(6)],
                       'mask_conj': [f'm{k}.png' for k in range(5)] + [None]})
    z = np.zeros((h, w), bool)

    def det(k, **kw):
        crop = np.zeros((h, w, 3), np.uint8)
        crop[0, 0, 0] = k                                              # метка снимка для заглушек
        return dict({'crop': crop, 'conj': z.copy(), 'sclera': z.copy(), 'use': {'conj'}, 'kind': 'глаз',
                     'notes': [], 'k': k}, **kw)

    dets = {os.path.join('R', f'i{k}.jpg'): det(k, incons='радужка не совпала' if k == 1 else '') for k in range(6)}
    raw = {0: conj, 2: z, 3: conj, 4: conj, 5: conj}
    route = {0: (conj, 'eyes_defy', ''), 1: (z, None, 'ткани не согласованы'), 2: (z, None, 'конъюнктива не найдена'),
             3: (conj, 'eyes_defy_sclera', ''), 4: (conj, 'eyes_defy', ''), 5: (conj, 'eyes_defy', '')}
    fold_of = dict(enumerate(df.fold))

    def conj_mask(seg, crop):
        k = int(crop[0, 0, 0])
        assert not seg.trap and seg.key == (fold_of[k],), f'утечка: снимок фолда {fold_of[k]} сегментатором {seg.key}'
        return raw[k].copy()

    def eye_route(d0, seg, limits):
        k = d0['k']
        assert seg.key == (fold_of[k],) and limits['fold'] == fold_of[k], 'не те сегментатор/пределы'
        m, key, why = route[k]
        return dict(d0, conj=m.copy(), use={'conj'} if m.any() else set()), key, why

    monkeypatch.setattr(EV, 'eye_route', eye_route)
    monkeypatch.setattr(EV.SB, 'conj_mask', conj_mask)
    monkeypatch.setattr(PR, '_inconsistent', lambda d: d['incons'])
    monkeypatch.setattr(EV, 'load_mask', lambda p: gt.copy())
    monkeypatch.setattr(EV, 'BAD_DOCTOR_MASKS', {'ED_India_4': 'испорчена'})
    segs = {(f,): _FoldSeg((f,)) for f in range(3)} | {(f, g): _FoldSeg((f, g), trap=True)
                                                       for f in range(3) for g in range(3) if f != g}
    limits = {k: dict(AN.LIMITS, fold=k[0]) for k in segs}
    iou0 = M.iou(conj, gt)
    J = [{'i': k, 'role': 'test', 'key': route[k][1], 'why': route[k][2],
          'iou': [iou0, 0.0, 0.0, 0.0, None, None][k]} for k in range(6)]
    J += [{'i': k, 'role': f, 'key': 'eyes_defy', 'why': '', 'iou': 0.123} for k in range(6) for f in range(3)
          if f != fold_of[k]]                                         # обучающие роли (nested) — не в отчёт
    return df, segs, limits, J, _DetCache(dets), iou0


def test_core_seg_records_statuses_and_folds(monkeypatch):
    df, segs, limits, J, cache, iou0 = _core_case(monkeypatch)
    recs = EV.core_seg_records('R', df, segs, limits, J, 1, cache)
    P = SM.records_frame(recs).set_index('id')
    assert list(P.index) == list(df.patient_id) and set(P.task) == {'core'}
    assert P.loc['ED_India_0', ['status', 'accepted']].tolist() == ['ok', True]
    assert P.loc['ED_India_0', 'iou'] == pytest.approx(iou0)
    assert P.loc['ED_India_1', 'status'] == 'refused' and P.loc['ED_India_1', 'pred_empty'] is None
    assert P.loc['ED_India_1', 'why'] == 'ткани не согласованы'
    assert P.loc['ED_Italy_2', 'status'] == 'ok' and P.loc['ED_Italy_2', 'pred_empty']     # пустая маска — ошибка
    assert P.loc['ED_Italy_2', 'iou'] == 0 and np.isnan(P.loc['ED_Italy_2', 'precision'])
    assert P.loc['ED_Italy_3', 'status'] == 'refused' and 'eyes_defy_sclera' in P.loc['ED_Italy_3', 'why']
    assert P.stage.tolist() == ['accepted', 'before_seg_inconsistent', 'empty_mask', 'no_conj_model', 'no_gt', 'no_gt']
    assert P.loc['ED_Italy_3', 'iou'] == pytest.approx(iou0)          # диагностика отклонённой маски
    assert P.loc['ED_India_4', 'status'] == 'no_gt' and P.loc['ED_Italy_5', 'status'] == 'no_gt'
    S = SM.summarize(SM.records_frame(recs)).set_index('country')
    assert (S.loc['all', 'n_ok'], S.loc['all', 'n_refused'], S.loc['all', 'n_no_gt']) == (2, 2, 2)
    assert S.loc['all', 'iou_mean'] == pytest.approx(iou0 / 2)


@pytest.mark.parametrize('case', ['iou', 'key', 'none', 'missing', 'segmenter'])
def test_core_seg_records_mismatch_stops(monkeypatch, case):
    df, segs, limits, J, cache, iou0 = _core_case(monkeypatch)
    if case == 'iou':
        J[0]['iou'] = iou0 + 1e-6
    elif case == 'key':
        J[2]['key'] = 'eyes_defy_conj'
    elif case == 'none':
        J[4]['iou'] = 0.0                                              # у снимка без эталона IoU быть не может
    elif case == 'missing':
        J = [x for x in J if not (x['i'] == 5 and x['role'] == 'test')]
    else:
        del segs[(1,)]
    with pytest.raises(EV.SegConsistencyError):
        EV.core_seg_records('R', df, segs, limits, J, 1, cache)


def test_core_seg_records_tolerance_is_explicit(monkeypatch):
    df, segs, limits, J, cache, iou0 = _core_case(monkeypatch)
    J[0]['iou'] = iou0 + EV.IOU_TOL / 10                                # округление в пределах допуска
    assert len(EV.core_seg_records('R', df, segs, limits, J, 1, cache)) == 6
    J[0]['iou'] = iou0 + 1e-7
    with pytest.raises(EV.SegConsistencyError, match='ED_India_0'):
        EV.core_seg_records('R', df, segs, limits, J, 1, cache, tol=1e-8)
    assert len(EV.core_seg_records('R', df, segs, limits, J, 1, cache, tol=1e-6)) == 6


def test_write_seg_metrics_stops_on_mismatch(tmp_path, capsys, monkeypatch):
    df, segs, limits, J, cache, _ = _core_case(monkeypatch)
    recs = EV.core_seg_records('R', df, segs, limits, J, 1, cache)
    meta = EV.write_seg_metrics(str(tmp_path), recs, 'eyes_defy: расхождение IoU')
    assert meta == {'status': 'stopped', 'why': 'eyes_defy: расхождение IoU'}
    assert not list(tmp_path.iterdir()) and 'НЕ записаны' in capsys.readouterr().out
    assert EV.write_seg_metrics(str(tmp_path), [], None) is None and not list(tmp_path.iterdir())
    meta = EV.write_seg_metrics(str(tmp_path), recs, None)
    assert meta['status'] == 'done' and meta['iou_tol'] == EV.IOU_TOL
    P = pd.read_csv(tmp_path / 'seg_per_image.csv')
    assert list(P.columns) == list(SM.CSV_COLUMNS) and len(P) == 6
    assert (tmp_path / 'seg_summary.csv').exists()


# ---- интеграция: настоящие eye_route / learned_conj / tissue_route / _core_job, подменён только SB.conj_mask ----
import cv2  # noqa: E402

from anemia.cache import empty_timings  # noqa: E402
from test_anatomy import MaskSeg, fake_det  # noqa: E402


class _LidSeg(MaskSeg):
    """Сегментатор фолда: его маску отдаёт подменённый SB.conj_mask (MaskSeg нужен для диагностики _seg_steps)."""
    def __init__(self, key, mask, trap=False):
        super().__init__(mask)
        self.key, self.m, self.trap = key, mask, trap


class _StepOneCache:
    """Шаг 1 (v12det) — готовые ткани; остальное (признаки v12seg) считается по-настоящему, без кэша."""
    def __init__(self, dets):
        self.dets = dets

    def fetch(self, kind, files, arrays, params, func, *args):
        if kind == 'v12det':
            return self.dets[files['image']], empty_timings()
        return EV.NO_CACHE.fetch(kind, files, arrays, params, func, *args)


def _integration_case(tmp_path, monkeypatch):
    """5 снимков, по одному в фолде (у каждого фолда свой сегментатор):
    0 — веко принято; 1 — сегментатор не нашёл веко; 2 — «пятно» (анатомия); 3 — шаг 1 несогласован (сегментатор
    не запускается); 4 — полоса века выше радужки (_inconsistent после сегментатора)."""
    gt = strip()
    lids = {0: strip(), 1: np.zeros_like(gt), 2: blob(120, 230, 40), 3: strip(), 4: strip(y=10)}
    ids = ['ED_India_10', 'ED_Italy_11', 'ED_India_12', 'ED_Italy_13', 'ED_India_14']
    df = pd.DataFrame({'dataset': 'eyes_defy', 'patient_id': ids, 'group': ids, 'fold': range(5),
                       'image': [f'i{k}.jpg' for k in range(5)], 'mask_conj': [f'm{k}.png' for k in range(5)],
                       'sex': ['F', 'M', 'F', 'M', 'F'], 'age': [30.0] * 5, 'pregnant': [None] * 5,
                       'hb_g_l': [110.0, 140.0, 125.0, 150.0, 100.0]})
    df['anemic'] = df.hb_g_l < 120
    m = np.zeros((*gt.shape, 3), np.uint8)
    m[gt] = (200, 60, 60)                                              # маска врача: ткань на чёрном фоне
    dets = {}
    for k in range(5):
        cv2.imwrite(str(tmp_path / f'm{k}.png'), m)
        d = fake_det()
        d['iris'], d['conj'] = (120.0, 130.0, 35.0), strip()              # шаг 1: радужка + пороговое веко
        if k == 3:
            sc = np.zeros_like(gt)
            sc[20:60, 20:200] = True
            d.update(iris=None, sclera=sc, use={'sclera'})               # только «склера» без радужки — отказ шага 1
        dets[os.path.join(str(tmp_path), f'i{k}.jpg')] = d
    segs = {(k,): _LidSeg((k,), lids[k], trap=k == 3) for k in range(5)}

    def conj_mask(seg, crop):
        assert not seg.trap, 'сегментатор запущен на снимке, отклонённом до сегментатора'
        return seg.m.copy()

    monkeypatch.setattr(EV.SB, 'conj_mask', conj_mask)
    return df, segs, _StepOneCache(dets)


@pytest.mark.parametrize('anatomy', [True, False])
def test_core_seg_records_match_real_route(tmp_path, monkeypatch, anatomy):
    df, segs, cache = _integration_case(tmp_path, monkeypatch)
    limits = {k: AN.LIMITS for k in segs} if anatomy else None
    J = EV.core_jobs(str(tmp_path), df, 1, segs, cache, limits=limits)       # настоящий _core_job
    recs = EV.core_seg_records(str(tmp_path), df, segs, limits, J, 1, cache)  # сверка IoU и модели внутри
    P = SM.records_frame(recs).set_index('id')
    route = {x['i']: x for x in J}
    assert route[0]['key'] == 'eyes_defy_conj' and route[0]['iou'] == pytest.approx(1.0)
    assert P.loc['ED_India_10', ['status', 'stage', 'accepted']].tolist() == ['ok', 'accepted', True]
    assert P.loc['ED_India_10', 'iou'] == pytest.approx(route[0]['iou'])
    assert P.loc['ED_Italy_11', ['status', 'stage']].tolist() == ['ok', 'empty_mask']
    assert P.loc['ED_Italy_11', 'iou'] == 0 and np.isnan(P.loc['ED_Italy_11', 'precision'])
    assert route[1]['key'] is None and 'сегментатор' in route[1]['why']
    assert P.loc['ED_Italy_13', ['status', 'stage']].tolist() == ['refused', 'before_seg_inconsistent']
    assert P.loc['ED_Italy_13', 'pred_empty'] is None
    assert P.loc['ED_India_14', ['status', 'stage']].tolist() == ['refused', 'after_seg_inconsistent']
    assert 'выше радужки' in P.loc['ED_India_14', 'why']
    if anatomy:
        assert P.loc['ED_India_12', ['status', 'stage']].tolist() == ['refused', 'anatomy']
        assert 'анатомическая проверка' in P.loc['ED_India_12', 'why'] and P.loc['ED_India_12', 'pred_empty'] is False
    else:
        assert P.loc['ED_India_12', 'stage'] != 'anatomy'                  # без анатомии этого этапа нет
    for k, x in route.items():                                             # статус следует из модели маршрута
        conj_used = x['key'] is not None and 'conj' in EV.CORE.SPECS[x['key']]['tissues']
        assert (P.iloc[k]['stage'] == 'accepted') == conj_used


def test_seg_pass_catches_errors():
    def boom(*a):
        raise ValueError('сломалось')

    def mismatch(*a):
        raise EV.SegConsistencyError('IoU разошёлся')

    assert EV.seg_pass('eyes_defy', lambda *a: [1, 2]) == ([1, 2], None)
    assert EV.seg_pass('eyes_defy', mismatch) == ([], ('stopped', 'eyes_defy: IoU разошёлся'))
    recs, err = EV.seg_pass('eyes_defy', boom)
    assert recs == [] and err[0] == 'failed' and 'ValueError: сломалось' in err[1]


def test_seg_ext_record_error_keeps_history_rows(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError('сбой отчёта')

    rows0 = None
    for broken in (False, True):
        with monkeypatch.context() as mp:
            if broken:
                mp.setattr(EV, '_seg_record', boom)
            rows, recs, err = _run_seg_ext_stub(mp)
        if not broken:
            rows0 = rows
            assert err is None and recs
        else:
            assert recs == [] and 'RuntimeError: сбой отчёта' in err
            assert pd.DataFrame(rows).equals(pd.DataFrame(rows0))           # строки истории те же


def _run_seg_ext_stub(mp):
    ids = [f'ED_India_{k}' for k in range(5)]
    df = pd.DataFrame({'patient_id': ids, 'group': ids, 'fold': range(5), 'image': ids, 'mask_conj': ids})
    idx = pd.DataFrame({'id': [f'phone_conj_masks_{k}' for k in range(4)], 'group': ['a', 'a', 'b', 'c'],
                        'file': ['x'] * 4, 'mask': ['y'] * 4})
    gt = strip()
    mp.setattr(EV.EXT, 'seg_index', lambda root: idx)
    mp.setattr(EV, '_segpix_job', lambda root, r, i, cache: (('ed', r['patient_id']), {}))
    mp.setattr(EV, 'external_seg_samples', lambda root, jobs, cache, idx: [('ex', i) for i in idx['id']])
    mp.setattr(EV, '_seg_test_item', lambda root, r, kind: (('ed', r['patient_id']) if kind == 'ed'
                                                              else ('ex', r['id']), gt))
    mp.setattr(EV.SB, 'fit_samples', _TrainedOn)
    mp.setattr(EV.SB, 'conj_mask', lambda model, tag: strip(x0=40, x1=200))
    mp.setattr(EV.AN, 'fit_limits', lambda stats: AN.LIMITS)
    return EV.run_seg_external('root', df, 1)


@pytest.mark.parametrize('where', ['summarize', 'print_seg_summary'])
def test_write_seg_metrics_failure_leaves_no_files(tmp_path, monkeypatch, capsys, where):
    df, segs, limits, J, cache, _ = _core_case(monkeypatch)
    recs = EV.core_seg_records('R', df, segs, limits, J, 1, cache)

    def boom(*a):
        raise KeyError('колонка')

    # до записи файлов (сводка) и после записи обоих (печать) — в обоих случаях файлов не остаётся
    monkeypatch.setattr(EV.SM if where == 'summarize' else EV, where, boom)
    meta = EV.write_seg_metrics(str(tmp_path), recs)
    assert meta['status'] == 'failed' and 'KeyError' in meta['why']
    assert not list(tmp_path.iterdir()) and 'НЕ записаны' in capsys.readouterr().out
    meta = EV.write_seg_metrics(str(tmp_path), recs, 'core: второй проход: ValueError', 'failed')
    assert meta == {'status': 'failed', 'why': 'core: второй проход: ValueError'} and not list(tmp_path.iterdir())
