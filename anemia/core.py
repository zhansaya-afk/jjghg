"""Ядро HemaVision: итоговая модель риска анемии по фото и её проверка на фиксированных фолдах манифеста.

Признаки — как в v12 (anemia.v12_model.eye_features): конъюнктива c_, склера s_, конъюнктива после баланса белого
по склере w_ (+ пол у взрослых). Склера не нашлась — её признаки NaN (HGB умеет пропуски), конъюнктива обязательна.
Это вариант «v12 conj+sclera» из scripts/experiment_accuracy.py, а не новая гипотеза; цифры — evaluate.py --task core.

Ответ — три уровня риска по P(анемия) с двумя порогами (anemia.selective.zone_thresholds):
    высокий — среди таких ответов на обучающих пациентах ≥ ZONE_TARGET больных;
    низкий  — среди таких ≥ ZONE_TARGET здоровых;
    средний — между ними: «не уверен, сдайте анализ крови».
Пороги, допуск OOD и разрешение показывать Hb подбираются ТОЛЬКО по вложенной групповой CV на тех данных, что
переданы в fit_core. cross_validate вызывает fit_core на обучающих фолдах и применяет к тестовому фолду, поэтому
порог никогда не видит тестовых пациентов; итоговая модель (scripts/train_core.py) — тот же fit_core на всех данных.

Hb (г/л) — экспериментально: регрессия HGB на тех же признаках. Диапазон = прогноз ± 80-й перцентиль |ошибки|
на вложенной CV. Показывается, только если на вложенной CV регрессия ошибается хотя бы на HB_MIN_SKILL меньше
константы (медианы Hb обучающих пациентов) — иначе цифра Hb ничего не добавляет к уровню риска.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from . import metrics as M
from . import v12_model as V12
from .selective import zone_thresholds

ZONE_TARGET = 0.90       # доля верных ответов среди «высокий» / «низкий» на обучающих пациентах
SENS_TARGET = 0.80       # второй, бинарный порог — на чувствительность 0,80 (как v12.1), для сравнения
HB_MIN_SKILL = 0.10      # Hb показываем, если MAE модели ≤ 0,9 × MAE константы на вложенной CV
HB_RANGE_Q = 0.80        # полуширина диапазона Hb — этот перцентиль |ошибки| на вложенной CV
OOD_MAX = 0.30           # как в v12: больше 30% признаков далеко вне обучающего диапазона — снимок «не такой»
INNER_FOLDS = 4
RISK_LEVELS = ('низкий', 'средний', 'высокий')

SPECS = {   # какие признаки у какой модели: взрослые — снимок глаза, дети — уже вырезанная конъюнктива
    'eyes_defy': {'prefixes': ('c_', 's_', 'w_'), 'sex': True, 'title': 'взрослые (Eyes-defy-anemia)'},
    'cp_anemic': {'prefixes': ('c_',), 'sex': False, 'title': 'дети 6–59 мес (CP-AnemiC)'},
}


def feature_list(columns, dataset):
    s = SPECS[dataset]
    f = [k for k in columns if k.startswith(s['prefixes'])]
    return f + (['male'] if s['sex'] else [])


def usable(F):
    """Строки пациентов, у которых конъюнктива нашлась на исходном снимке: без неё модель не применяется."""
    c = [k for k in F.columns if k.startswith('c_')]
    ok = set(F[F.aug == 0].dropna(subset=c).group)
    return F[F.group.isin(ok)].dropna(subset=c)


# ------------------------------------------------------------------ части модели
def _fit_cls(T, feats, seed=0):
    m = V12.fit(T[feats].values, T.y.values, seed)
    return m, V12.class_prior(T[T.aug == 0].y.values)


def _p_anemia(m, prior, X):
    return 1 - V12.predict(m, np.asarray(X, float), prior)[:, 0]


def _hb_model():
    return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300, l2_regularization=1.0,
                                         min_samples_leaf=15, random_state=0)


def ood_share(x, lo, hi):
    """Доля признаков далеко за пределами обучающих (за 1–99 перцентилем плюс четверть размаха); NaN не считаются."""
    x = np.atleast_2d(np.asarray(x, float))
    span = np.maximum(hi - lo, 1e-6)
    out = (x < lo - 0.25 * span) | (x > hi + 0.25 * span)
    n = np.isfinite(x).sum(1)
    return np.where(n > 0, out.sum(1) / np.maximum(n, 1), 1.0)


def risk_level(p, t_lo, t_hi):
    """0 — низкий, 1 — средний, 2 — высокий."""
    p = np.asarray(p, float)
    return np.where(p >= t_hi, 2, np.where(p <= t_lo, 0, 1))


# ------------------------------------------------------------------ обучение
def fit_core(F, feats, dataset, seed=0, inner_folds=INNER_FOLDS, zone_target=ZONE_TARGET):
    """F: строки признаков (aug 0 — исходный снимок, aug > 0 — обучающие копии при другом свете), колонки
    group, anemic, y (класс ВОЗ 0/1/2), hb_g_l. → словарь модели (всё, что нужно для anemia.predict)."""
    F = F.reset_index(drop=True)
    O = F[F.aug == 0].reset_index(drop=True)
    anem, hb = O.anemic.values.astype(bool), O.hb_g_l.values.astype(float)
    p_in, hb_in, hb0_in = np.zeros(len(O)), np.zeros(len(O)), np.zeros(len(O))
    for itr, ite in GroupKFold(min(inner_folds, O.group.nunique())).split(O, groups=O.group):
        T = F[F.group.isin(set(O.group.iloc[itr]))]          # копии при другом свете — только пациентов train
        m, prior = _fit_cls(T, feats, seed)
        p_in[ite] = _p_anemia(m, prior, O[feats].values[ite])
        hb_in[ite] = _hb_model().fit(T[feats].values, T.hb_g_l.values).predict(O[feats].values[ite])
        hb0_in[ite] = np.median(hb[itr])
    t_lo, t_hi = zone_thresholds(p_in, anem, zone_target)
    mae, mae0 = M.mae(hb_in, hb), M.mae(hb0_in, hb)
    model, prior = _fit_cls(F, feats, seed)
    X0 = O[feats].values.astype(float)
    return {
        'dataset': dataset, 'title': SPECS[dataset]['title'], 'feats': list(feats),
        'n_patients': int(O.group.nunique()), 'n_anemic': int(anem.sum()),
        'cls': model, 'prior': prior, 'hb': _hb_model().fit(F[feats].values, F.hb_g_l.values),
        't_lo': t_lo, 't_hi': t_hi, 'zone_target': zone_target,
        'thr_sens': float(M.threshold_for_sensitivity(p_in, anem, SENS_TARGET)),
        'hb_mae_cv': float(mae), 'hb_mae_const_cv': float(mae0), 'hb_skill_cv': float(1 - mae / mae0),
        'hb_ok': bool(1 - mae / mae0 >= HB_MIN_SKILL),
        'hb_halfwidth': float(np.quantile(np.abs(hb_in - hb), HB_RANGE_Q)),
        'ood_lo': np.nanpercentile(X0, 1, axis=0), 'ood_hi': np.nanpercentile(X0, 99, axis=0),
    }


def apply_core(B, X):
    """→ P(анемия), уровень риска 0/1/2, прогноз Hb (г/л), доля признаков вне обучающего диапазона."""
    X = np.atleast_2d(np.asarray(X, float))
    p = _p_anemia(B['cls'], B['prior'], X)
    return p, risk_level(p, B['t_lo'], B['t_hi']), B['hb'].predict(X), ood_share(X, B['ood_lo'], B['ood_hi'])


# ------------------------------------------------------------------ проверка на фолдах манифеста
def cross_validate(F, feats, dataset, S=None, seed=0):
    """OOF по фолдам манифеста: модель и все её пороги обучаются на 4 фолдах, применяются к пятому.
    S — признаки тех же снимков при 8 стресс-условиях света (group, cond): риск по ним считает модель того фолда,
    где пациент был тестовым. → (OOF-таблица исходных снимков, стресс-таблица или None)."""
    O = F[F.aug == 0].reset_index(drop=True)
    cols = {k: np.full(len(O), np.nan) for k in ('p', 'level', 'hb_pred', 'ood', 'hb_ok', 'thr_sens', 'hb_hw')}
    stress = []
    for f in sorted(O.fold.unique()):
        te = np.flatnonzero(O.fold.values == f)
        B = fit_core(F[F.fold != f], feats, dataset, seed)
        p, lv, hbp, ood = apply_core(B, O[feats].values[te])
        for k, v in (('p', p), ('level', lv), ('hb_pred', hbp), ('ood', ood)):
            cols[k][te] = v
        cols['hb_ok'][te], cols['thr_sens'][te], cols['hb_hw'][te] = B['hb_ok'], B['thr_sens'], B['hb_halfwidth']
        if S is not None and len(S):
            St = S[S.group.isin(set(O.group.values[te]))]
            if len(St):
                ps, ls, _, _ = apply_core(B, St[feats].values)
                stress.append(St[['group', 'cond', 'anemic']].assign(p=ps, level=ls))
    return O.assign(**cols), (pd.concat(stress, ignore_index=True) if stress else None)


def core_metrics(O, S=None):
    """Метрики OOF ядра: AUC, три уровня риска, бинарный порог на чувствительность 0,80, Hb, OOD, свет."""
    anem, p, lv, g = O.anemic.values.astype(bool), O.p.values, O.level.values.astype(int), O.group.values
    r = {'n': len(O), 'n_anemic': int(anem.sum())}
    if 0 < anem.sum() < len(anem):
        r['auc'] = M.auc(p, anem)
        r['auc_lo'], r['auc_hi'] = M.bootstrap_ci(M.auc, g, 1000, 0, p, anem)
    for i, name in enumerate(('low', 'mid', 'high')):
        r[f'share_{name}'] = float((lv == i).mean())
    sure = lv != 1
    r['acc_sure'] = float(((lv == 2) == anem)[sure].mean()) if sure.any() else float('nan')
    r['high_ppv'], r['high_ppv_lo'], r['high_ppv_hi'] = M.wilson(int((anem & (lv == 2)).sum()), int((lv == 2).sum()))
    r['low_npv'], r['low_npv_lo'], r['low_npv_hi'] = M.wilson(int((~anem & (lv == 0)).sum()), int((lv == 0).sum()))
    r['missed_in_low'] = int((anem & (lv == 0)).sum())
    flag = p >= O.thr_sens.values
    r['sens'], r['sens_lo'], r['sens_hi'] = M.wilson(int((flag & anem).sum()), int(anem.sum()))
    r['spec'], r['spec_lo'], r['spec_hi'] = M.wilson(int((~flag & ~anem).sum()), int((~anem).sum()))
    r['mae'] = M.mae(O.hb_pred.values, O.hb_g_l.values)
    r['hb_shown'] = float(O.hb_ok.mean())
    r['hb_range_cover'] = float((np.abs(O.hb_pred.values - O.hb_g_l.values) <= O.hb_hw.values).mean())
    r['ood_share'] = float((O.ood.values > OOD_MAX).mean())
    if S is not None and len(S):
        r['light_level_change'] = float(S.groupby('group')['level'].nunique().gt(1).mean())
        per_cond = [M.auc(S[S.cond == c].p.values, S[S.cond == c].anemic.values.astype(bool))
                    for c in S.cond.unique()]
        r['auc_light'] = float(np.nanmean(per_cond))
    return r
