"""Модели Hb по признакам цвета и честная кросс-валидация по фиксированным фолдам манифеста.

Риск-скор = норма ВОЗ пациента − предсказанный Hb (г/л): больше — выше риск. Порог скора подбирается
ВНУТРИ обучающих фолдов (вложенная CV) на чувствительность ≥ target и затем применяется к тестовому фолду.
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .metrics import threshold_for_sensitivity


def make_model(kind):
    if kind == 'hgb':
        return HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=300,
                                             l2_regularization=1.0, min_samples_leaf=15, random_state=0)
    if kind == 'ridge':
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 4, 25)))
    raise ValueError(kind)


def fit_predict(kind, X_tr, y_tr, X_te):
    return make_model(kind).fit(X_tr, y_tr).predict(X_te)


def cross_validate(F, feats, kind='hgb', target_sens=0.80, inner_folds=4, stress=None):
    """F: таблица признаков; строки с aug == 0 — исходные снимки, aug > 0 — обучающие копии при другом свете.
    Обязательные колонки: group, fold, hb_g_l, normal_thr, aug. stress: таблица признаков стресс-теста
    (group, cond) — предсказывается моделью того фолда, где пациент был тестовым.
    Возвращает OOF-таблицу исходных снимков: pred, score, thr, flag (+ stress_pred по условиям)."""
    O = F[F.aug == 0].copy()
    O['pred'] = np.nan
    O['thr'] = np.nan
    stress_out = []
    for f in sorted(O.fold.unique()):
        tr_all = F[(F.fold != f)]
        te = O.index[O.fold == f]
        model = make_model(kind).fit(tr_all[feats].values, tr_all.hb_g_l.values)
        O.loc[te, 'pred'] = model.predict(O.loc[te, feats].values)
        if stress is not None:
            S = stress[stress.group.isin(O.loc[te, 'group'])]
            if len(S):
                stress_out.append(S.assign(pred=model.predict(S[feats].values))[['group', 'cond', 'pred']])
        # вложенная CV на обучающих фолдах → OOF-скор на train → порог
        tr0 = O[O.fold != f]
        inner = np.full(len(tr0), np.nan)
        k = min(inner_folds, tr0.group.nunique())
        for itr, ite in GroupKFold(k).split(tr0, groups=tr0.group):
            g_tr = set(tr0.group.iloc[itr])
            T = tr_all[tr_all.group.isin(g_tr)]
            inner[ite] = make_model(kind).fit(T[feats].values, T.hb_g_l.values).predict(tr0[feats].values[ite])
        s_tr = tr0.normal_thr.values - inner
        O.loc[te, 'thr'] = threshold_for_sensitivity(s_tr, tr0.anemic.values, target_sens)
    O['score'] = O.normal_thr - O.pred
    O['flag'] = O.score >= O.thr
    import pandas as pd
    return O, (pd.concat(stress_out) if stress_out else None)
