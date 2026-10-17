"""Эксперименты «как поднять точность v12» на Eyes-defy, на тех же фолдах и признаках, что error_analysis.py.

    python scripts/error_analysis.py --data data --tag ea_v12        # один раз: считает и кэширует признаки
    python scripts/experiment_accuracy.py --data data --tag ea_v12

Модели (маска конъюнктивы — автоматическая v12, как в демо):
  v12 conj        — как в блокноте: 3 класса ВОЗ по цвету конъюнктивы;
  v12 conj+sclera — + склера и конъюнктива после баланса белого по склере (пропуски склеры — NaN, HGB их понимает);
  relative        — + разности «конъюнктива − склера» по каждому признаку цвета;
  hb-reg          — регрессия Hb (г/л) на признаках relative, класс — по нормам ВОЗ пациента.
Для каждой: AUC (всего, Индия, Италия), сбалансированная точность 3 классов, чувств./спец. при пороге 0,80,
и зона «не уверен» с порогами на 90% и 95% точности уверенных ответов. Все пороги — вложенной CV на train.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anemia import metrics as M  # noqa: E402
from anemia import v12_model as V12  # noqa: E402
from anemia.selective import zone_metrics, zone_thresholds  # noqa: E402

TARGETS = (0.90, 0.95)


def load(data, tag, source='auto'):
    res = pd.read_pickle(os.path.join(data, 'results', f'ea_cache_{tag}.pkl'))
    F = pd.DataFrame([x for rows, _, _, _ in res for x in rows])
    info = pd.DataFrame([x[2] for x in res])
    F = F[F.source == source].merge(info[['group', 'country', 'hb_g_l', 'sex', 'age', 'doc_diff']], on='group')
    F = F[F.doc_diff <= 25]                                  # те же пациенты, что в error_analysis
    c = [k for k in F.columns if k.startswith('c_')]
    F = F[F.group.isin(F[F.aug == 0].dropna(subset=c).group)]
    for k in c:                                              # «конъюнктива относительно склеры»
        s = 's_' + k[2:]
        if s in F.columns:
            F['d_' + k[2:]] = F[k] - F[s]
    thr = [V12.who_thresholds(None if s == '?' else s, a if a == a else 30.0) for s, a in zip(F.sex, F.age)]
    F['norm_thr'] = [t[0] * 10 for t in thr]
    F['mod_thr'] = [t[1] * 10 for t in thr]
    return F.reset_index(drop=True)


def feature_sets(F):
    c = [k for k in F.columns if k.startswith('c_')] + ['male']
    s = [k for k in F.columns if k.startswith(('s_', 'w_'))]
    d = [k for k in F.columns if k.startswith('d_')]
    return {'v12 conj': ('cls', c), 'v12 conj+sclera': ('cls', c + s), 'relative': ('cls', c + s + d),
            'hb-reg': ('reg', c + s + d)}


def _reg():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                         min_samples_leaf=15, random_state=0)


def fit_predict(kind, feats, tr, te):
    """→ (P(анемия)-подобный скор, класс 0/1/2, прогноз Hb или NaN) для строк te (aug 0)."""
    X, Xt = tr[feats].values, te[feats].values
    if kind == 'cls':
        tr0 = tr[tr.aug == 0]
        P = V12.predict(V12.fit(X, tr.y.values), Xt, V12.class_prior(tr0.y.values))
        return 1 - P[:, 0], P.argmax(1), np.full(len(te), np.nan)
    hb = _reg().fit(X, tr.hb_g_l.values).predict(Xt)
    k = np.where(hb >= te.norm_thr.values, 0, np.where(hb >= te.mod_thr.values, 1, 2))
    return te.norm_thr.values - hb, k, hb


def run(F, kind, feats, inner_folds=4):
    O = F[F.aug == 0].reset_index(drop=True)
    score, k3, hb = np.zeros(len(O)), np.zeros(len(O), int), np.full(len(O), np.nan)
    thr80 = np.zeros(len(O))
    zones = {t: np.zeros((len(O), 2)) for t in TARGETS}
    for f in sorted(O.fold.unique()):
        te = np.flatnonzero(O.fold.values == f)
        tr = F[F.fold != f]
        score[te], k3[te], hb[te] = fit_predict(kind, feats, tr, O.iloc[te])
        tr0 = O[O.fold != f].reset_index(drop=True)
        inner = np.zeros(len(tr0))
        for itr, ite in GroupKFold(inner_folds).split(tr0, groups=tr0.group):
            T = tr[tr.group.isin(set(tr0.group.iloc[itr]))]
            inner[ite] = fit_predict(kind, feats, T, tr0.iloc[ite])[0]
        thr80[te] = M.threshold_for_sensitivity(inner, tr0.anemic.values, 0.80)
        for t in TARGETS:
            zones[t][te] = zone_thresholds(inner, tr0.anemic.values, t)
    return O.assign(score=score, k3=k3, hb_pred=hb, thr80=thr80), zones


def summarize(name, O, zones):
    y, anem, s = O.y.values, O.anemic.values.astype(bool), O.score.values
    flag = s >= O.thr80.values
    r = {'модель': name, 'n': len(O), 'AUC': M.auc(s, anem)}
    for c in ('India', 'Italy'):
        m = (O.country == c).values
        r[f'AUC {c}'] = M.auc(s[m], anem[m])
    r['верный класс (сбаланс.)'] = balanced_accuracy_score(y, O.k3.values)
    r['чувств. @0.80'], r['спец. @0.80'] = flag[anem].mean(), (~flag[~anem]).mean()
    r['да/нет верно @0.80'] = (flag == anem).mean()
    if O.hb_pred.notna().any():
        r['MAE Hb'] = M.mae(O.hb_pred.values, O.hb_g_l.values)
    for t, Z in zones.items():
        z = zone_metrics(s, anem, Z[:, 0], Z[:, 1])
        p = int(t * 100)
        r[f'зона{p}: уверенных'] = z['coverage']
        r[f'зона{p}: верно'] = z['acc_sure']
        r[f'зона{p}: чувств.'] = z['sens_sure']
        r[f'зона{p}: спец.'] = z['spec_sure']
        r[f'зона{p}: пропущено больных'] = z['missed']
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('ANEMIA_DATA', 'data'))
    ap.add_argument('--tag', default='ea_v12')
    ap.add_argument('--out', default=os.path.join(ROOT, 'results', 'errors'))
    a = ap.parse_args()
    F = load(a.data, a.tag)
    rows = []
    for name, (kind, feats) in feature_sets(F).items():
        print(f'{name}: {len(feats)} признаков...', flush=True)
        O, zones = run(F, kind, feats)
        rows.append(summarize(name, O, zones))
    R = pd.DataFrame(rows)
    path = os.path.join(a.out, a.tag, 'experiments.csv')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    R.to_csv(path, index=False, float_format='%.3f')
    with pd.option_context('display.width', 250, 'display.max_columns', 40, 'display.float_format', '{:.3f}'.format):
        print(R.set_index('модель').T)
    print(f'\n→ {path}')


if __name__ == '__main__':
    main()
