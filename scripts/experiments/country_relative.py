"""Эксперимент 2026-11-12: ложный риск у здоровых индийцев (19/26 FP) — две гипотезы против baseline ядра.

    A  baseline   — признаки ядра eyes_defy (c_ + s_ + w_ + пол), обучение как в anemia.core.fit_core.
    B  страна     — то же, но строки обучения каждой модели (вложенные и итоговая) пересэмплированы так, что в каждой
                    стране половина больных: доля анемии больше не выдаёт страну (country_balance).
    C  относит.   — только относительные признаки: d_ = конъюнктива − склера (та же статистика того же канала) и w_
                    (конъюнктива после баланса белого по склере) + пол; абсолютных c_ / s_ нет.
    D  B + C.

Всё остальное как в ядре и не подгоняется: калибровка Platt, пороги «низкий» (чувств. 0,80) и «высокий» (≥ 90% больных),
допуск Hb — тем же правилом по вложенной групповой CV только на обучающих пациентах (вложенные OOF — по исходным, не
пересэмплированным пациентам). Фолды — манифеста; обучающие признаки фолда f — маски сегментаторов, не видевших f
(как evaluate --task core). Все варианты — на одних пациентах: принятых моделью обеих тканей (у остальных склеры нет,
относительные признаки не считаются — их доля печатается отдельно). Повторы — seed 0..n−1 (повтор редких строк и
пересэмплирование стран). Production-код, модель Я1 и пороги не меняются; это проверка на внутренней CV Eyes-defy,
а не доказательство клинической эффективности.

    python scripts/experiments/country_relative.py --data "$ANEMIA_DATA" --out "$ANEMIA_DATA/results/exp_country_relative"
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from anemia import core as CORE
from anemia import metrics as M
from anemia.selective import zone_thresholds

KEY = 'eyes_defy'
VARIANTS = {'A_baseline': {'rel': False, 'balance': False},
            'B_страна': {'rel': False, 'balance': True},
            'C_относит': {'rel': True, 'balance': False},
            'D_страна+относит': {'rel': True, 'balance': True}}


# ------------------------------------------------------------------ признаки и пересэмплирование
def country(groups):
    """Страна Eyes-defy из группы пациента (ED_<страна>_<номер>)."""
    s = pd.Series(np.asarray(groups, dtype=str)).str.split('_').str[1]
    return s.set_axis(groups.index) if isinstance(groups, pd.Series) else s


def add_relative(F):
    """d_<стат> = c_<стат> − s_<стат> для каждой статистики, что есть у обеих тканей; склеры нет → NaN."""
    pairs = [k[2:] for k in F.columns if k.startswith('c_') and 's_' + k[2:] in F.columns]
    return F.assign(**{'d_' + k: F['c_' + k] - F['s_' + k] for k in pairs})


def relative_feature_list(columns):
    return [k for k in columns if k.startswith(('d_', 'w_'))] + ['male']


def relative_ok(F):
    """Относительные признаки посчитаны полностью (склера есть и все разности — числа)."""
    f = [k for k in F.columns if k.startswith(('d_', 'w_'))]
    return np.isfinite(F[f].to_numpy(float)).all(1)


def country_balance(T, seed=0):
    """Строки обучения → каждая ячейка страна × анемия дополнена повтором своих же строк до размера самой большой:
    в каждой стране доля анемии 0,5, страны равны. Ни один пациент не добавлен и не потерян."""
    c = country(T.group).values
    cells = {(cc, a): np.flatnonzero((c == cc) & (T.anemic.values == a)) for cc in np.unique(c) for a in (False, True)}
    empty = sorted({cc for (cc, a), ix in cells.items() if not len(ix)})
    if empty:
        raise ValueError(f'в стране {", ".join(empty)} нет одного из классов — уравнять нельзя')
    rng = np.random.default_rng(seed)
    n = max(len(ix) for ix in cells.values())
    idx = np.concatenate([np.concatenate([ix, rng.choice(ix, n - len(ix), replace=True)]) for ix in cells.values()])
    return T.iloc[idx].reset_index(drop=True)


# ------------------------------------------------------------------ модель (fit_core с пересэмплированием обучения)
def fit_variant(F, feats, key, balance, seed=0, inner_folds=CORE.INNER_FOLDS):
    """Копия CORE.fit_core; отличие одно — balance=True пересэмплирует строки, на которых учатся классификатор и
    регрессия Hb (во вложенной CV и в итоговой модели). Пороги, калибровка, Hb-допуск, OOD — по исходным пациентам.
    balance=False даёт ровно CORE.fit_core (тест)."""
    prep = (lambda T: country_balance(T, seed)) if balance else (lambda T: T)
    F = F.reset_index(drop=True)
    O = F[F.aug == 0].reset_index(drop=True)
    anem, hb = O.anemic.values.astype(bool), O.hb_g_l.values.astype(float)
    p_in, hb_in, hb0_in = np.zeros(len(O)), np.zeros(len(O)), np.zeros(len(O))
    for itr, ite in GroupKFold(min(inner_folds, O.group.nunique())).split(O, groups=O.group):
        T = prep(F[F.group.isin(set(O.group.iloc[itr]))])
        m, prior = CORE._fit_cls(T, feats, seed)
        if balance:                                  # частота классов для поправки — исходных пациентов, как в ядре
            prior = CORE.V12.class_prior(O.y.values[itr])
        p_in[ite] = CORE._p_anemia(m, prior, O[feats].values[ite])
        hb_in[ite] = CORE._hb_model().fit(T[feats].values, T.hb_g_l.values).predict(O[feats].values[ite])
        hb0_in[ite] = np.median(hb[itr])
    cal = CORE.fit_platt(p_in, anem)
    t_lo = float(M.threshold_for_sensitivity(p_in, anem, CORE.LOW_SENS_TARGET))
    t_hi = max(zone_thresholds(p_in, anem, CORE.ZONE_TARGET)[1], t_lo)
    t_lo, t_hi = (float(CORE.calibrate(cal, t)) for t in (t_lo, t_hi))
    mae, mae0 = M.mae(hb_in, hb), M.mae(hb0_in, hb)
    TF = prep(F)
    model, prior = CORE._fit_cls(TF, feats, seed)
    if balance:
        prior = CORE.V12.class_prior(O.y.values)
    X0 = O[feats].values.astype(float)
    return {'dataset': key, 'title': CORE.SPECS[key]['title'], 'tissues': CORE.SPECS[key]['tissues'], 'feats': list(feats),
            'n_patients': int(O.group.nunique()), 'n_anemic': int(anem.sum()), 'cls': model, 'prior': prior,
            'hb': CORE._hb_model().fit(TF[feats].values, TF.hb_g_l.values), 'cal': cal, 't_lo': t_lo, 't_hi': t_hi,
            'zone_target': CORE.ZONE_TARGET, 'low_sens_target': CORE.LOW_SENS_TARGET,
            'thr_sens': float(CORE.calibrate(cal, M.threshold_for_sensitivity(p_in, anem, CORE.SENS_TARGET))),
            'hb_mae_cv': float(mae), 'hb_mae_const_cv': float(mae0), 'hb_skill_cv': float(1 - mae / mae0),
            'hb_ok': bool(1 - mae / mae0 >= CORE.HB_MIN_SKILL),
            'hb_halfwidth': float(np.quantile(np.abs(hb_in - hb), CORE.HB_RANGE_Q)),
            'ood_lo': np.nanpercentile(X0, 1, axis=0), 'ood_hi': np.nanpercentile(X0, 99, axis=0)}


def cv(F, feats, key, train_by_fold, balance, seed=0, S=None):
    """OOF по фолдам манифеста (как CORE.cross_validate) → таблица исходных снимков (+ уровень при 8 условиях
    света в колонке-списке не храним: стресс → отдельная таблица в атрибуте attrs['stress'])."""
    O = F[F.aug == 0].reset_index(drop=True)
    out, stress = [], []
    for f in sorted(O.fold.unique()):
        te = O[O.fold == f]
        T = train_by_fold[f]
        assert not set(T.group) & set(te.group), 'утечка: пациент теста в обучении'
        B = fit_variant(T, feats, key, balance, seed)
        p, lv, hbp, ood = CORE.apply_core(B, te[feats].values)
        if 'who_normal' in te:
            lv, _ = CORE.reconcile(lv, hbp, B['hb_halfwidth'], B['hb_ok'], te.who_normal.values)
        out.append(te[['group', 'fold', 'anemic', 'hb_g_l']].assign(p=p, level=lv, hb_pred=hbp, ood=ood,
                                                                   accepted=ood <= CORE.OOD_MAX))
        if S is not None and len(S):
            St = S[S.group.isin(set(te.group))]
            if len(St):
                _, ls, hbs, _ = CORE.apply_core(B, St[feats].values)
                if 'who_normal' in St:
                    ls, _ = CORE.reconcile(ls, hbs, B['hb_halfwidth'], B['hb_ok'], St.who_normal.values)
                stress.append(St[['group', 'cond']].assign(level=ls))
    R = pd.concat(out, ignore_index=True)
    R.attrs['stress'] = pd.concat(stress, ignore_index=True) if stress else None
    return R


# ------------------------------------------------------------------ метрики
def sens(lv, y):
    y = np.asarray(y, bool)
    return float((np.asarray(lv)[y] >= 1).mean()) if y.any() else float('nan')


def spec(lv, y):
    y = np.asarray(y, bool)
    return float((np.asarray(lv)[~y] == 0).mean()) if (~y).any() else float('nan')


def _auc(p, y):
    y = np.asarray(y, bool)
    return M.auc(p, y) if 0 < y.sum() < len(y) else float('nan')


def paired_delta(fn, groups, a, b, y, n_boot=1000, seed=0):
    """Δ = fn(b) − fn(a) на одних пациентах + 95% ДИ парного бутстрэпа по пациентам + число отброшенных повторов
    (в повторе нет нужного класса, метрика NaN). Отброшенные сужают ДИ — verdict тогда не принимает вариант."""
    groups, a, b, y = (np.asarray(v) for v in (groups, a, b, y))
    d0 = fn(b, y) - fn(a, y)
    ug = np.unique(groups)
    pos = {g: np.flatnonzero(groups == g) for g in ug}
    rng = np.random.default_rng(seed)
    ds = []
    for _ in range(n_boot):
        ix = np.concatenate([pos[g] for g in rng.choice(ug, len(ug), replace=True)])
        v = fn(b[ix], y[ix]) - fn(a[ix], y[ix])
        if np.isfinite(v):
            ds.append(v)
    lo, hi = (np.percentile(ds, [2.5, 97.5]) if ds else (np.nan, np.nan))
    return float(d0), float(lo), float(hi), int(n_boot - len(ds))   # последнее — отброшено повторов (метрика не определена)


def summary(O, n_boot=1000):
    A = O[O.accepted]
    y, lv, g = A.anemic.values.astype(bool), A.level.values.astype(int), A.group.values
    err = A.hb_pred.values - A.hb_g_l.values
    r = {'n': len(A), 'n_ood': int((~O.accepted).sum()), 'анемия': int(y.sum()), 'здоровых': int((~y).sum())}
    r['AUC'] = _auc(A.p.values, y)
    r['AUC_lo'], r['AUC_hi'] = M.bootstrap_ci(_auc, g, n_boot, 0, A.p.values, y)
    r['чувств'], r['чувств_lo'], r['чувств_hi'] = M.wilson(int((y & (lv >= 1)).sum()), int(y.sum()))
    r['спец'], r['спец_lo'], r['спец_hi'] = M.wilson(int((~y & (lv == 0)).sum()), int((~y).sum()))
    r['FP'], r['FP_высокий'], r['FN'] = int((~y & (lv >= 1)).sum()), int((~y & (lv == 2)).sum()), int((y & (lv == 0)).sum())
    r['MAE_Hb'] = float(np.abs(err).mean())
    r['MAE_lo'], r['MAE_hi'] = M.bootstrap_ci(lambda e: float(np.abs(e).mean()), g, n_boot, 0, err)
    r['сдвиг_Hb_здоровых'] = float(err[~y].mean()) if (~y).any() else float('nan')
    S = O.attrs.get('stress')
    if S is not None and len(S):
        S = S[S.group.isin(set(A.group))]
        r['свет_уровень_меняется'] = float(S.groupby('group')['level'].nunique().gt(1).mean())
    return r


# ------------------------------------------------------------------ решение (правило задано до запуска)
SENS_DROP_MAX = 0.05     # чувствительность в целом не ниже baseline больше чем на 5 п.п. (точка) ...
SENS_LO_MIN = -0.15      # ... и нижняя граница ДИ Δ не ниже −15 п.п.
AUC_LO_MIN = -0.03       # AUC в целом не хуже baseline: нижняя граница ДИ Δ ≥ −0,03
INDIA_HEALTHY_MIN = 30   # меньше здоровых индийцев в общем наборе → итог не выше «не доказано»


def verdict(D, seeds):
    """D — парные Δ к baseline (seed 0), seeds — метрики по seed. Принять вариант, только если:
    1) специфичность в Индии выросла: нижняя граница ДИ Δ > 0 (главная цель);
    2) FP снизился не ценой FN: Δ чувств. ≥ −SENS_DROP_MAX и её ДИ ≥ SENS_LO_MIN;
    3) AUC в целом не упал (ДИ Δ ≥ AUC_LO_MIN);
    4) снижение FP больше разброса seed: max FP варианта < min FP baseline.
    Ограничение размера: здоровых индийцев < INDIA_HEALTHY_MIN или есть отброшенные бутстрэп-повторы Δ спец. Индии →
    не выше «не доказано» (отказ по FN / AUC остаётся отказом)."""
    out = {}
    base = seeds[seeds['вариант'] == 'A_baseline'].FP
    for v in [x for x in D['вариант'].unique()]:
        d = D[D['вариант'] == v].set_index(['часть', 'метрика'])
        sp, se, au = d.loc[('India', 'спец')], d.loc[('все', 'чувств')], d.loc[('все', 'AUC')]
        fp = seeds[seeds['вариант'] == v].FP
        why = []
        if se['Δ'] < -SENS_DROP_MAX or se['lo'] < SENS_LO_MIN:
            why.append(f'FN растёт: Δ чувств. {100 * se["Δ"]:+.0f} п.п. [{100 * se["lo"]:+.0f}…{100 * se["hi"]:+.0f}]')
        if au['lo'] < AUC_LO_MIN:
            why.append(f'AUC может падать: Δ {au["Δ"]:+.3f} [{au["lo"]:+.3f}…{au["hi"]:+.3f}]')
        nh, drop = int(sp['n_здоровых']), int(sp['отброшено_повторов'])
        if why:
            out[v] = 'отклонить: ' + '; '.join(why)
        elif nh < INDIA_HEALTHY_MIN:
            out[v] = (f'не доказано: здоровых в Индии n={nh} < {INDIA_HEALTHY_MIN} — подгруппа мала для вывода '
                      f'(Δ спец. Индии {100 * sp["Δ"]:+.0f} п.п. [{100 * sp["lo"]:+.0f}…{100 * sp["hi"]:+.0f}], '
                      f'отброшено {drop} бутстрэп-повторов)')
        elif drop > 0:
            out[v] = (f'не доказано: отброшено {drop} бутстрэп-повторов без здоровых индийцев — ДИ Δ спец. Индии '
                      f'[{100 * sp["lo"]:+.0f}…{100 * sp["hi"]:+.0f}] может быть занижен по ширине')
        elif sp['lo'] <= 0:
            out[v] = (f'не доказано: Δ спец. Индии {100 * sp["Δ"]:+.0f} п.п. [{100 * sp["lo"]:+.0f}…'
                      f'{100 * sp["hi"]:+.0f}] — ДИ задевает 0')
        elif fp.max() >= base.min():
            out[v] = f'не доказано: FP по seed {fp.min()}–{fp.max()} перекрывается с baseline {base.min()}–{base.max()}'
        else:
            out[v] = (f'принять как кандидата для проверки: Δ спец. Индии {100 * sp["Δ"]:+.0f} п.п. '
                      f'[{100 * sp["lo"]:+.0f}…{100 * sp["hi"]:+.0f}], Δ чувств. {100 * se["Δ"]:+.0f} п.п.')
    return out


# ------------------------------------------------------------------ охват: кто и почему не вошёл в сравнение
def coverage(n_manifest, J, runs, common):
    """n_manifest — пациентов Eyes-defy в манифесте; J — задания core_jobs; runs — {(вариант, seed): (OOF, n признаков)};
    common — общий набор сравнения. → dict для coverage.json: маршруты и причины отказа, исключённые по каждому
    варианту и seed (вне маршрута модели обеих тканей + OOD), общий набор, выбывшие из-за OOD других вариантов."""
    test = [x for x in J if x['role'] == 'test']
    routes = pd.Series([x['key'] or 'отказ' for x in test]).value_counts()
    why = pd.Series([str(x['why']).split(':')[0].strip() or 'без причины' for x in test if x['key'] is None])
    n_route = int(routes.get(KEY, 0))
    c = {'пациентов в манифесте': int(n_manifest), 'модель обеих тканей (eyes_defy)': n_route,
         'только конъюнктива (склеры нет → относительных признаков нет)': int(routes.get('eyes_defy_conj', 0)),
         'только склера': int(routes.get('eyes_defy_sclera', 0)), 'отказ маршрута': int(routes.get('отказ', 0)),
         'причины отказа маршрута': {k: int(v) for k, v in why.value_counts().items()}}
    per = {}
    for (v, seed), (O, _) in runs.items():
        acc = int(O.accepted.sum())
        ex = n_manifest - acc
        per[f'{v} seed {seed}'] = {'вне маршрута модели обеих тканей': int(n_manifest - len(O)),
                                   'OOD': int((~O.accepted).sum()), 'исключено всего': int(ex),
                                   'доля исключённых': ex / n_manifest if n_manifest else float('nan'),
                                   'принято': acc}
    c['по вариантам и seed'] = per
    c['общий набор'] = {'n': len(common), 'доля от манифеста': len(common) / n_manifest if n_manifest else float('nan')}
    out = {}
    for v in dict.fromkeys(k[0] for k in runs):
        own = set.intersection(*(set(O.group[O.accepted]) for (vv, _), (O, _) in runs.items() if vv == v))
        out[v] = len(own - set(common))
    c['исключено из общего набора из-за OOD других вариантов'] = out
    return c


# ------------------------------------------------------------------ запуск в Colab
def build(root, jobs):
    import evaluate as EV
    man = pd.read_csv(os.path.join(root, 'manifest.csv'))
    df = man[man.dataset == 'eyes_defy'].reset_index(drop=True)
    cache = EV.FeatureCache(os.path.join(root, 'cache', 'features'), 'on',
                            salt={'evaluate_py': EV.source_version(os.path.abspath(EV.__file__))})
    segs = EV.conj_segmenters(root, df, jobs, cache, None, nested=True, full=False, external=True)
    # пределы анатомии — по маскам врачей только обучающих фолдов каждого теста, как evaluate --task core
    lims = EV.anatomy_limits(root, df, jobs, list(segs), cache, external=True)
    J = EV.core_jobs(root, df, jobs, segs, cache, None, nested=True, limits=lims)
    return EV, df, J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('ANEMIA_DATA', 'data'))
    ap.add_argument('--out', required=True)
    ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--n-boot', type=int, default=1000)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    import subgroup_report as SR
    EV, df, J = build(a.data, a.jobs)
    print(f'[{time.time() - t0:.0f} с] признаки готовы (маски сегментаторов, не видевших тестовый фолд)', flush=True)
    F = add_relative(CORE.usable(EV._frame(J, 'test', KEY), KEY))
    S = EV._frame(J, 'test', KEY, s=True)
    S = add_relative(S[S.group.isin(set(F.group))]) if len(S) else None
    folds = sorted(set(df.fold.astype(int)))
    train = {f: add_relative(CORE.usable(EV._frame(J, f), KEY)) for f in folds}
    O0 = F[F.aug == 0]
    ok = relative_ok(O0)
    man = df.assign(group=df.group)
    runs = {}
    for vname, v in VARIANTS.items():
        feats = relative_feature_list(F.columns) if v['rel'] else CORE.feature_list(F.columns, KEY)
        for seed in range(a.seeds):
            runs[vname, seed] = (cv(F, feats, KEY, train, v['balance'], seed, S), len(feats))
            print(f'[{time.time() - t0:.0f} с] {vname} seed {seed} обучен', flush=True)
    # сравнение — только на пациентах, которых приняли все варианты и все seed (OOD у разных признаков разный)
    common = set.intersection(*(set(O.group[O.accepted]) for O, _ in runs.values()))
    cover = coverage(len(df), J, runs, common)
    cover['eyes_defy: относительные признаки посчитаны полностью'] = int(ok.sum())
    cover['eyes_defy: часть разностей NaN (HGB умеет пропуски)'] = int((~ok).sum())
    cover['доля пациентов без полных относительных признаков'] = round(1 - int(ok.sum()) / len(df), 3)
    print(json.dumps(cover, ensure_ascii=False, indent=1))
    rows, oof0 = [], {}
    for (vname, seed), (O, nf) in runs.items():
        O.to_csv(os.path.join(a.out, f'oof_{vname}_seed{seed}.csv'), index=False)
        C = O[O.group.isin(common)].assign(accepted=True)
        C.attrs['stress'] = O.attrs.get('stress')
        rows.append({'вариант': vname, 'seed': seed, 'n_признаков': nf, 'n_ood_варианта': int((~O.accepted).sum()),
                     **summary(C, a.n_boot)})
        if seed == 0:
            oof0[vname] = C
            SR.subgroup_table(C, man).to_csv(os.path.join(a.out, f'subgroups_{vname}.csv'), index=False)
        print(f'{vname} seed {seed}: AUC {rows[-1]["AUC"]:.3f}, FP {rows[-1]["FP"]}, FN {rows[-1]["FN"]}', flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(os.path.join(a.out, 'summary_by_seed.csv'), index=False)
    # парные Δ к baseline (seed 0, одни пациенты), всего и по странам
    base = oof0['A_baseline'].set_index('group')
    dl = []
    for vname, O in oof0.items():
        if vname == 'A_baseline':
            continue
        O = O.set_index('group').loc[base.index]
        keep = (base.accepted & O.accepted).values
        for part in ('все', 'India', 'Italy'):
            m = keep & ((country(base.index) == part).values if part != 'все' else True)
            g, y = base.index.values[m], base.anemic.values[m].astype(bool)
            for name, fn, xa, xb in (('спец', spec, base.level.values[m], O.level.values[m]),
                                     ('чувств', sens, base.level.values[m], O.level.values[m]),
                                     ('AUC', _auc, base.p.values[m], O.p.values[m])):
                d, lo, hi, drop = paired_delta(fn, g, xa, xb, y, a.n_boot)
                dl.append({'вариант': vname, 'часть': part, 'метрика': name, 'Δ': d, 'lo': lo, 'hi': hi, 'n': int(m.sum()),
                           'n_здоровых': int((~y).sum()), 'n_анемия': int(y.sum()), 'отброшено_повторов': drop})
    D = pd.DataFrame(dl)
    D.to_csv(os.path.join(a.out, 'paired_delta_vs_baseline.csv'), index=False)
    agg = R.groupby('вариант', sort=False).agg(**{k: (k, 'mean') for k in ('AUC', 'чувств', 'спец', 'FP', 'FN', 'MAE_Hb')},
                                               AUC_sd=('AUC', 'std'), FP_min=('FP', 'min'), FP_max=('FP', 'max'),
                                               FN_min=('FN', 'min'), FN_max=('FN', 'max'))
    agg.to_csv(os.path.join(a.out, 'summary_mean.csv'))
    s0 = R[R.seed == 0].set_index('вариант')
    print('\nSeed 0 (ДИ: AUC и MAE — бутстрэп по пациентам, доли — Уилсон); среднее по seed — summary_mean.csv')
    print('| Вариант | n | AUC [ДИ] | Чувств., % [ДИ] | Спец., % [ДИ] | FP (высокий) | FN | MAE Hb [ДИ] | свет меняет уровень, % |')
    print('|---|---|---|---|---|---|---|---|---|')
    for v, r in s0.iterrows():
        print(f"| {v} | {r.n} | {r.AUC:.3f} [{r.AUC_lo:.3f}–{r.AUC_hi:.3f}] | {100 * r['чувств']:.0f} "
              f"[{100 * r['чувств_lo']:.0f}–{100 * r['чувств_hi']:.0f}] | {100 * r['спец']:.0f} [{100 * r['спец_lo']:.0f}–"
              f"{100 * r['спец_hi']:.0f}] | {r.FP} ({r['FP_высокий']}) | {r.FN} | {r.MAE_Hb:.1f} [{r.MAE_lo:.1f}–{r.MAE_hi:.1f}] | "
              f"{100 * r.get('свет_уровень_меняется', np.nan):.0f} |")
    print('\nСреднее по seed (разброс FP/FN — min–max):')
    print(agg.round(3).to_string())
    print('\nПарные Δ к baseline (seed 0):')
    print(D.round(3).to_string(index=False))
    V = verdict(D, R)
    print('\nРешение (правило задано до запуска, см. verdict):')
    for k, t in V.items():
        print(f'  {k}: {t}')
    cover['решение'] = V
    for vname in VARIANTS:
        T = pd.read_csv(os.path.join(a.out, f'subgroups_{vname}.csv'))
        print(f'\n{vname} — подгруппы:')
        print(T[['группа', 'оценено', 'анемия', 'здоровых', 'FP_средний_плюс', 'FN_низкий', 'чувств', 'спец', 'AUC',
                 'MAE_Hb', 'сдвиг_Hb_здоровых']].round(3).to_string(index=False))
    with open(os.path.join(a.out, 'coverage.json'), 'w') as fh:
        json.dump(cover, fh, ensure_ascii=False, indent=1)
    print(f'\n[{time.time() - t0:.0f} с] → {a.out}')


if __name__ == '__main__':
    main()
