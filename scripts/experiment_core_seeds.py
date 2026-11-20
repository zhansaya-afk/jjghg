"""Эксперимент «ансамбль по seed» для модели риска ядра (Eyes-defy, те же фолды, что evaluate.py --task core).

Гипотеза (задана ДО запуска, 2026-11-20): ответ модели риска заметно зависит от случайного seed повтора редких
классов (V12.fit): в Colab-эксперименте 2026-11-12 (коммит cad82ea, 3 seed, 182 пациента; общая папка проекта
exp_country_relative_2026-11-12/РЕЗУЛЬТАТ.md, в metrics_history.csv не записан) baseline дал FP 18–23 и FN 16–21
при одном и том же коде. Среднее P(анемия) по 5
классификаторам с разными seed (внутри fit_core: и во вложенной CV порогов/калибровки, и в итоговой модели) должно
уменьшить этот разброс и не ухудшить AUC.

Что сравнивается (одни и те же маски, маршрут, отказы и признаки; меняется только классификатор риска):
    A<s>  — текущая модель (один классификатор) с seed s = 0…4; A0 = модель продукта;
    E<r>  — ансамбль из 5 классификаторов с seed 100 + 5r … 100 + 5r + 4, r = 0…4 (пять повторов ансамбля; seed не
            пересекаются с A, поэтому повторы E независимы от A).
Модель Hb, сегментация, QC/OOD и пороги-цели (чувствительность 0,80, точность зон 0,90) не меняются.

Правило решения (заранее, по модели «обе ткани»; модель «только конъюнктива» — для сведения, n мала):
    ПРИНЯТЬ ансамбль, если одновременно:
    1) средний AUC трёх повторов E ≥ среднего AUC пяти A − 0,005 и нижняя граница парного 95% ДИ ΔAUC(E0 − A0)
       > −0,02;
    2) средний AUC внутри стран E ≥ среднего у A − 0,005;
    3) разброс между повторами меньше: размах (max − min) FP + FN у пяти E меньше, чем у пяти A (поровну повторов);
    4) доля пациентов со сменой уровня при другом свете у E (среднее) не выше, чем у A (среднее) + 0,02;
    5) чувствительность «средний + высокий» у E (среднее) не ниже, чем у A (среднее) − 0,02.
    Иначе — отклонить (модель продукта остаётся A0). Улучшение AUC не заявляется, если ДИ ΔAUC содержит 0.

Запуск (Colab, данные Eyes-defy; ~75 мин признаки + ~30–40 мин модели):
    python scripts/experiment_core_seeds.py --data <data> [--quick]
--quick — 2 seed A и 1 повтор E (проверка, что код работает; решения не даёт).
Результат: <data>/results/experiments/core_seeds_<время>/ (summary.csv, paired.csv, oof_*.csv, decision.md).
Пациентские строки (OOF) — только в папку данных, не в репозиторий.
"""
import argparse
import contextlib
import datetime as dt
import json
import os
import sys
import time

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import evaluate as EV  # noqa: E402

from anemia import core as CORE  # noqa: E402
from anemia import metrics as M  # noqa: E402
from anemia import v12_model as V12  # noqa: E402
from anemia.cache import FeatureCache, source_version  # noqa: E402

N_ENS = 5
ENS_SEED0 = 100        # seed ансамблей не пересекаются с seed A0–A4
NAN = float('nan')
KEYS = ('eyes_defy', 'eyes_defy_conj')


@contextlib.contextmanager
def ensemble(n):
    """Внутри блока fit_core обучает n классификаторов (seed ENS_SEED0 + seed·n + k, k < n) и усредняет P(анемия)."""
    fit0, p0 = CORE._fit_cls, CORE._p_anemia

    def fit(T, feats, seed=0):
        ms = [V12.fit(T[feats].values, T.y.values, ENS_SEED0 + seed * n + k) for k in range(n)]
        return ms, V12.class_prior(T[T.aug == 0].y.values)

    def p(m, prior, X):
        return np.mean([p0(mi, prior, X) for mi in m], 0) if isinstance(m, list) else p0(m, prior, X)

    CORE._fit_cls, CORE._p_anemia = fit, p
    try:
        yield
    finally:
        CORE._fit_cls, CORE._p_anemia = fit0, p0


def run_variant(df, J, key, seed, n_ens):
    """Как evaluate.run_core, но с seed и (n_ens > 1) ансамблем → OOF (+ accepted) и стресс-таблица."""
    F = EV._frame(J, 'test', key)
    if not len(F):
        return None, None
    F = CORE.usable(F, key)
    feats = CORE.feature_list(F.columns, key)
    S = EV._frame(J, 'test', key, s=True)
    S = S[S.group.isin(set(F.group))] if len(S) else None
    train = {f: CORE.usable(EV._frame(J, f), key) for f in sorted(set(df.fold.astype(int)))}
    ctx = ensemble(n_ens) if n_ens > 1 else contextlib.nullcontext()
    with ctx:
        O, SP = CORE.cross_validate(F, feats, key, S, seed=seed, train_by_fold=train)
    return O.assign(accepted=O.ood.values <= CORE.OOD_MAX), SP


def metrics(O, SP, country):
    """Метрики на принятых снимках: всё из core_metrics + PR-AUC, AUC внутри стран и по странам, смещение Hb."""
    A = O[O.accepted]
    r = CORE.core_metrics(A, None if SP is None else SP[SP.group.isin(set(A.group))])
    y = A.anemic.values.astype(bool)
    r['pr_auc'] = float(average_precision_score(y, A.p.values)) if 0 < y.sum() < len(y) else float('nan')
    r['pr_auc_base'] = float(y.mean())
    c = A.group.map(country).values
    for row in M.auc_by_stratum(A.p.values, y, c, A.group.values, n_boot=1000):
        name = {'все': 'all', 'внутри стран': 'within'}.get(row['подвыборка'], row['подвыборка'])
        r[f'auc_{name}'], r[f'auc_{name}_lo'], r[f'auc_{name}_hi'] = row['auc'], row['auc_lo'], row['auc_hi']
        r[f'n_{name}'], r[f'n_anemic_{name}'] = row['n'], row['n_anemic']
    d = A.hb_pred.values - A.hb_g_l.values
    low = A.hb_g_l.values < 100
    r.update(hb_bias=float(d.mean()), hb_bias_low=float(d[low].mean()) if low.any() else float('nan'),
             n_hb_low=int(low.sum()), fp_fn=int(r['fp'] + r['fn']), n_ood=int((~O.accepted).sum()))
    return r


def paired(Oa, Ob, country, n_boot=1000):
    """ΔAUC и Δ AUC внутри стран (b − a) на общих принятых пациентах, парный бутстрэп по пациентам."""
    m = Oa.accepted.values & Ob.accepted.values
    y = Oa.anemic.values[m].astype(bool)
    pa, pb, g = Oa.p.values[m], Ob.p.values[m], Oa.group.values[m]
    c = pd.Series(g).map(country).values
    out = {'n_common': int(m.sum())}
    if not 0 < y.sum() < len(y):                 # один класс: AUC не определён
        return {**out, **{f'd_{k}{e}': NAN for k in ('auc', 'auc_within') for e in ('', '_lo', '_hi')}}
    for name, fn in (('auc', lambda a, b, t, s: M.auc(b, t) - M.auc(a, t)),
                     ('auc_within', lambda a, b, t, s: M.within_strata_auc(b, t, s) - M.within_strata_auc(a, t, s))):
        try:
            out[f'd_{name}'] = fn(pa, pb, y, c)
        except ValueError:                       # нет пар «больной — здоровый» внутри одной страны
            out[f'd_{name}'], out[f'd_{name}_lo'], out[f'd_{name}_hi'] = NAN, NAN, NAN
            continue
        out[f'd_{name}_lo'], out[f'd_{name}_hi'] = M.bootstrap_ci(fn, g, n_boot, 0, pa, pb, y, c)
    return out


def decide(sm, pr):
    """Правило из docstring модуля → (решение, пояснения по пунктам)."""
    a, e = sm[sm.variant.str.startswith('A')], sm[sm.variant.str.startswith('E')]
    if len(a) < 3 or len(e) < 2:
        return 'нет решения (--quick: мало повторов)', []
    rules = [
        ('1 AUC', e.auc.mean() >= a.auc.mean() - 0.005 and pr['d_auc_lo'] > -0.02,
         f'E {e.auc.mean():.3f} vs A {a.auc.mean():.3f}; ΔAUC(E0−A0) {pr["d_auc"]:+.3f} '
         f'[{pr["d_auc_lo"]:+.3f}; {pr["d_auc_hi"]:+.3f}]'),
        ('2 AUC внутри стран', e.auc_within.mean() >= a.auc_within.mean() - 0.005,
         f'E {e.auc_within.mean():.3f} vs A {a.auc_within.mean():.3f}'),
        ('3 разброс FP+FN', np.ptp(e.fp_fn) < np.ptp(a.fp_fn),
         f'E {e.fp_fn.min()}–{e.fp_fn.max()} vs A {a.fp_fn.min()}–{a.fp_fn.max()}'),
        ('4 свет', e.light_level_change.mean() <= a.light_level_change.mean() + 0.02,
         f'E {e.light_level_change.mean():.0%} vs A {a.light_level_change.mean():.0%}'),
        ('5 чувств. «средний+высокий»', e.zsens.mean() >= a.zsens.mean() - 0.02,
         f'E {e.zsens.mean():.0%} vs A {a.zsens.mean():.0%}'),
    ]
    ok = all(x for _, x, _ in rules)
    return ('ПРИНЯТЬ ансамбль' if ok else 'ОТКЛОНИТЬ (остаётся A0)'), rules


COLS = ['variant', 'model', 'n', 'n_anemic', 'auc', 'auc_lo', 'auc_hi', 'auc_within', 'auc_within_lo', 'auc_within_hi',
        'auc_India', 'n_India', 'auc_Italy', 'n_Italy', 'sens', 'spec', 'tp', 'fp', 'fn', 'tn', 'fp_fn', 'bacc2',
        'pr_auc', 'pr_auc_base', 'zsens', 'zspec', 'missed_in_low', 'share_low', 'share_mid', 'share_high', 'mae',
        'hb_bias', 'hb_bias_low', 'n_hb_low', 'hb_range_cover', 'ece', 'brier', 'light_level_change', 'n_ood']


def main():
    warnings.filterwarnings('ignore', message='Only one class is present')   # подвыборки с одним классом → NaN
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', default=os.environ.get('ANEMIA_DATA', 'data'))
    ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--quick', action='store_true', help='2 seed A + 1 повтор E: проверка кода, без решения')
    ap.add_argument('--no-external-seg', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    out = os.path.join(a.data, 'results', 'experiments', 'core_seeds_' + dt.datetime.now().strftime('%Y%m%d_%H%M%S'))
    os.makedirs(out, exist_ok=True)
    git = EV.git_state()
    print(f'experiment_core_seeds · commit {git["commit"]} · → {out}', flush=True)
    man = pd.read_csv(os.path.join(a.data, 'manifest.csv'))
    df = man[man.dataset == 'eyes_defy'].reset_index(drop=True)
    cache = FeatureCache(os.path.join(a.data, 'cache', 'features'), 'on',
                         salt={'evaluate_py': source_version(os.path.join(ROOT, 'scripts', 'evaluate.py'))})
    ext = not a.no_external_seg
    print(f'[{time.time() - t0:5.0f} с] сегментаторы века (без тестового фолда), пределы анатомии, маршрут и признаки '
          '— как evaluate.py --task core...', flush=True)
    segs = EV.conj_segmenters(a.data, df, a.jobs, cache, None, nested=True, full=False, external=ext)
    lims = EV.anatomy_limits(a.data, df, a.jobs, list(segs), cache, external=ext)
    J = EV.core_jobs(a.data, df, a.jobs, segs, cache, None, nested=True, limits=lims)
    rc = EV.route_counts(J, len(df))
    print('маршрут: ' + ', '.join(f'{k} {v}' for k, v in rc.items()), flush=True)
    per_group = pd.Series(EV.country_of(df).values, index=df.group.values).groupby(level=0).agg(set)
    mixed = set(per_group.index[per_group.map(len) > 1])      # копия снимка в обеих странах: как в auc_by_country
    if mixed:
        print(f'⚠️ группы в обеих странах исключены из всех метрик: {sorted(mixed)}', flush=True)
        J = [{**x, 'rows': [r for r in x['rows'] if r['group'] not in mixed],
              'srows': [r for r in x['srows'] if r['group'] not in mixed]} for x in J]
    country = per_group.map(lambda c: next(iter(c)))
    seeds_a = range(2 if a.quick else 5)
    reps_e = range(1 if a.quick else 5)
    rows, oof, pairs = [], {}, []
    for key in KEYS:
        for name, seed, n in ([(f'A{s}', s, 1) for s in seeds_a] + [(f'E{r}', r, N_ENS) for r in reps_e]):
            t1 = time.time()
            O, SP = run_variant(df, J, key, seed, n)
            if O is None:
                continue
            oof[(key, name)] = O
            r = metrics(O, SP, country)
            rows.append({'variant': name, 'model': key, **r})
            O[['group', 'fold', 'hb_g_l', 'anemic', 'p', 'level', 'hb_pred', 'thr_sens', 'ood', 'accepted']].to_csv(
                os.path.join(out, f'oof_{key}_{name}.csv'), index=False)
            print(f'[{time.time() - t0:5.0f} с] {key} {name}: n {r["n"]}, AUC {r.get("auc", NAN):.3f}, внутри стран '
                  f'{r.get("auc_within", NAN):.3f}, FP {r["fp"]} FN {r["fn"]}, свет {r.get("light_level_change", NAN):.0%} '
                  f'({time.time() - t1:.0f} с)', flush=True)
        if (key, 'A0') in oof and (key, 'E0') in oof:
            pairs.append({'model': key, 'pair': 'E0 − A0', **paired(oof[(key, 'A0')], oof[(key, 'E0')], country)})
    sm = pd.DataFrame(rows)
    sm = sm.reindex(columns=list(dict.fromkeys(list(sm.columns) + COLS)))   # нет метрики (один класс) → NaN
    sm[[c for c in COLS if c in sm]].to_csv(os.path.join(out, 'summary.csv'), index=False)
    pr = pd.DataFrame(pairs)
    pr.to_csv(os.path.join(out, 'paired.csv'), index=False)
    pd.set_option('display.width', 250)
    pd.set_option('display.max_columns', 40)
    show = ['variant', 'n', 'n_anemic', 'auc', 'auc_within', 'auc_India', 'auc_Italy', 'sens', 'spec', 'tp', 'fp',
            'fn', 'tn', 'bacc2', 'pr_auc', 'mae', 'hb_bias', 'hb_bias_low', 'light_level_change']
    lines = [f'# Эксперимент «ансамбль по seed» · коммит {git["commit"]} · {dt.date.today()}', '',
             'маршрут: ' + ', '.join(f'{k} {v}' for k, v in rc.items()), '']
    for key in KEYS:
        s = sm[sm.model == key]
        if not len(s):
            continue
        print(f'\n{key}:\n' + s[show].round(3).to_string(index=False), flush=True)
        lines += [f'## {key}', '', '```', s[show].round(3).to_string(index=False), '```', '']
    for p in pairs:
        line = (f"{p['model']} · {p['pair']} на {p['n_common']} общих пациентах: ΔAUC {p['d_auc']:+.3f} "
                f"[{p['d_auc_lo']:+.3f}; {p['d_auc_hi']:+.3f}], Δ AUC внутри стран {p['d_auc_within']:+.3f} "
                f"[{p['d_auc_within_lo']:+.3f}; {p['d_auc_within_hi']:+.3f}]")
        print(line)
        lines.append('- ' + line)
    main_pair = next((p for p in pairs if p['model'] == 'eyes_defy'), None)
    if main_pair is not None:
        verdict, rules = decide(sm[sm.model == 'eyes_defy'], main_pair)
        print(f'\nРЕШЕНИЕ (модель «обе ткани», правило задано до запуска): {verdict}')
        lines += ['', f'## Решение: {verdict}', '']
        for name, ok, txt in rules:
            print(f'  {"✅" if ok else "❌"} {name}: {txt}')
            lines.append(f'- {"✅" if ok else "❌"} {name}: {txt}')
    lines += ['', f'Время: {time.time() - t0:.0f} с. Ограничения: одна внутренняя CV Eyes-defy (одна насадка, две '
              'страны), не внешняя проверка; n «только конъюнктива» мало.']
    with open(os.path.join(out, 'decision.md'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines) + '\n')
    with open(os.path.join(out, 'run.json'), 'w') as f:
        json.dump({'git': git, 'args': vars(a), 'routes': rc, 'seconds': round(time.time() - t0)}, f, ensure_ascii=False,
                  indent=1, default=str)
    print(f'\nГотово за {time.time() - t0:.0f} с → {out}')


if __name__ == '__main__':
    main()
