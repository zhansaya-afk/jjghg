"""Ядро HemaVision: итоговая модель риска анемии по фото и её проверка на фиксированных фолдах манифеста.

Признаки — как в v12 (anemia.v12_model.eye_features): конъюнктива c_, склера s_, конъюнктива после баланса белого
по склере w_ (+ пол у взрослых). Склера не нашлась — её признаки NaN (HGB умеет пропуски), конъюнктива обязательна.
Это вариант «v12 conj+sclera» из scripts/experiment_accuracy.py, а не новая гипотеза; цифры — evaluate.py --task core.

P(анемия) калибрована: Platt (логистическая регрессия на logit P) по вложенным OOF-вероятностям обучающих
пациентов. Сырые вероятности HGB переуверены (обучение на копиях при другом свете и с повтором редких классов),
поэтому без калибровки P = 0,97 соответствовала меньшей доле больных, а порог «высокий» уходил к 0,99.
Калибровка монотонна: какие пациенты в какой зоне, она не меняет — меняет только цифру P и цифры порогов.

Ответ — три уровня риска по калиброванной P(анемия) с двумя порогами (подбор по вложенным OOF-вероятностям):
    низкий  — P < t_lo: t_lo = наибольший порог, при котором «средний + высокий» находят ≥ LOW_SENS_TARGET больных
              обучающих пациентов (цель для невиданных: чувствительность ≥ 0,80, специфичность «низкого» ≥ 0,70 —
              вторая не подбирается, а проверяется evaluate.py);
    высокий — P ≥ t_hi: среди таких ответов на обучающих пациентах ≥ ZONE_TARGET больных (anemia.selective);
    средний — между ними: «не уверен, сдайте анализ крови».
Согласование с Hb (reconcile): если Hb показывается и весь его диапазон ниже нормы ВОЗ пациента, уровень не ниже
«высокого» — диапазон = прогноз ± 80-й перцентиль |ошибки|, так что его верхняя граница ниже нормы примерно у 90%
таких пациентов, это тот же критерий, что у «высокого». Обратного правила нет (скрининг: не понижаем риск).
Пороги, калибровка, допуск OOD и разрешение показывать Hb подбираются ТОЛЬКО по вложенной групповой CV на тех данных, что
переданы в fit_core. cross_validate вызывает fit_core на обучающих фолдах и применяет к тестовому фолду, поэтому
порог никогда не видит тестовых пациентов; итоговая модель (scripts/train_core.py) — тот же fit_core на всех данных.

Hb (г/л) — экспериментально: регрессия HGB на тех же признаках. Диапазон = прогноз ± 80-й перцентиль |ошибки|
на вложенной CV. Показывается, только если на вложенной CV регрессия ошибается хотя бы на HB_MIN_SKILL меньше
константы (медианы Hb обучающих пациентов) — иначе цифра Hb ничего не добавляет к уровню риска.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from . import metrics as M
from . import v12_model as V12
from .selective import zone_thresholds

ZONE_TARGET = 0.90       # доля верных ответов среди «высокий» / «низкий» на обучающих пациентах
SENS_TARGET = 0.80       # бинарный порог на чувствительность 0,80 (как v12.1) = граница «низкий» / «средний»
LOW_SENS_TARGET = SENS_TARGET   # «средний + высокий» находят ≥ 80% больных (вложенная CV на train)
LOW_SPEC_GOAL = 0.70     # цель для специфичности «низкого» — проверяется evaluate.py, не подбирается
CAL_EPS = 1e-4           # P обрезается в [eps, 1 − eps] перед logit
HB_MIN_SKILL = 0.10      # Hb показываем, если MAE модели ≤ 0,9 × MAE константы на вложенной CV
HB_RANGE_Q = 0.80        # полуширина диапазона Hb — этот перцентиль |ошибки| на вложенной CV
OOD_MAX = 0.30           # как в v12: больше 30% признаков далеко вне обучающего диапазона — снимок «не такой»
INNER_FOLDS = 4
RISK_LEVELS = ('низкий', 'средний', 'высокий')

SPECS = {   # модель → датасет манифеста, признаки, какие ткани обязательны у снимка (снимок глаза взрослого;
    # детской модели нет — CP-AnemiC исключён 2026-10-19). Отдельная модель на каждый набор найденных тканей:
    # недостающая ткань не заменяется нулями или признаками другой ткани
    'eyes_defy': {'data': 'eyes_defy', 'prefixes': ('c_', 's_', 'w_'), 'need': ('c_',), 'sex': True,
                  'tissues': ('conj', 'sclera'), 'title': 'взрослые (Eyes-defy-anemia), конъюнктива + склера'},
    'eyes_defy_conj': {'data': 'eyes_defy', 'prefixes': ('c_',), 'need': ('c_',), 'sex': True,
                       'tissues': ('conj',), 'title': 'взрослые (Eyes-defy-anemia), только конъюнктива'},
    'eyes_defy_sclera': {'data': 'eyes_defy', 'prefixes': ('s_',), 'need': ('s_',), 'sex': True,
                         'tissues': ('sclera',), 'title': 'взрослые (Eyes-defy-anemia), только склера'},
}


def specs_for(dataset):
    """Модели, которые обучаются на этом датасете манифеста."""
    return [k for k, s in SPECS.items() if s['data'] == dataset]


def feature_list(columns, key):
    s = SPECS[key]
    f = [k for k in columns if k.startswith(s['prefixes'])]
    return f + (['male'] if s['sex'] else [])


def usable(F, key=None):
    """Строки пациентов, у которых на исходном снимке нашлись ткани, обязательные для модели key
    (по умолчанию — конъюнктива): без них модель не применяется."""
    need = SPECS[key]['need'] if key else ('c_',)
    c = [k for k in F.columns if k.startswith(need)]
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
    """0 — низкий (P < t_lo), 1 — средний, 2 — высокий (P ≥ t_hi)."""
    p = np.asarray(p, float)
    return np.where(p >= t_hi, 2, np.where(p < t_lo, 0, 1))


# ------------------------------------------------------------------ калибровка
def _logit(p):
    p = np.clip(np.asarray(p, float), CAL_EPS, 1 - CAL_EPS)
    return np.log(p / (1 - p))


def fit_platt(p, y):
    """Platt: P_кал = σ(a·logit(P) + b) по вложенным OOF-вероятностям обучающих пациентов. → (a, b).
    Наклон a ≤ 0 (вырожденный случай: модель не различает классы) → без калибровки (1, 0)."""
    y = np.asarray(y).astype(int)
    if y.min() == y.max():
        return 1.0, 0.0
    lr = LogisticRegression(C=1e4, max_iter=1000).fit(_logit(p)[:, None], y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    return (a, b) if a > 0 else (1.0, 0.0)


def calibrate(cal, p):
    """cal = (a, b) или None (модель старше калибровки) → калиброванная P(анемия). Пороги inf/−inf не трогает."""
    p = np.asarray(p, float)
    if cal is None:
        return p
    a, b = cal
    out = 1 / (1 + np.exp(-(a * _logit(p) + b)))
    return np.where(np.isfinite(p), out, p)


# ------------------------------------------------------------------ согласование с Hb
def hb_interval(hb, hw):
    """Показываемый диапазон Hb, г/л: прогноз ± полуширина, наружу до кратного 5."""
    hb = np.asarray(hb, float)
    return 5 * np.floor((hb - hw) / 5), 5 * np.ceil((hb + hw) / 5)


def reconcile(level, hb_pred, hw, hb_ok, who_normal):
    """Уровень не ниже «высокого», если Hb показывается (hb_ok) и весь диапазон Hb ниже нормы ВОЗ пациента
    (верхняя граница < who_normal, г/л). → (уровень, поднят ли)."""
    level = np.asarray(level, int)
    _, hi = hb_interval(hb_pred, hw)
    up = np.asarray(hb_ok, bool) & (hi < np.asarray(who_normal, float)) & (level < 2)
    return np.where(up, 2, level), up


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
    cal = fit_platt(p_in, anem)                         # пороги — по рангам сырых P, затем в шкалу калиброванной
    t_lo = float(M.threshold_for_sensitivity(p_in, anem, LOW_SENS_TARGET))
    t_hi = max(zone_thresholds(p_in, anem, zone_target)[1], t_lo)
    t_lo, t_hi = (float(calibrate(cal, t)) for t in (t_lo, t_hi))
    mae, mae0 = M.mae(hb_in, hb), M.mae(hb0_in, hb)
    model, prior = _fit_cls(F, feats, seed)
    X0 = O[feats].values.astype(float)
    return {
        'dataset': dataset, 'title': SPECS[dataset]['title'], 'tissues': SPECS[dataset]['tissues'], 'feats': list(feats),
        'n_patients': int(O.group.nunique()), 'n_anemic': int(anem.sum()),
        'cls': model, 'prior': prior, 'hb': _hb_model().fit(F[feats].values, F.hb_g_l.values),
        'cal': cal, 't_lo': t_lo, 't_hi': t_hi, 'zone_target': zone_target, 'low_sens_target': LOW_SENS_TARGET,
        'thr_sens': float(calibrate(cal, M.threshold_for_sensitivity(p_in, anem, SENS_TARGET))),
        'hb_mae_cv': float(mae), 'hb_mae_const_cv': float(mae0), 'hb_skill_cv': float(1 - mae / mae0),
        'hb_ok': bool(1 - mae / mae0 >= HB_MIN_SKILL),
        'hb_halfwidth': float(np.quantile(np.abs(hb_in - hb), HB_RANGE_Q)),
        'ood_lo': np.nanpercentile(X0, 1, axis=0), 'ood_hi': np.nanpercentile(X0, 99, axis=0),
    }


def apply_core(B, X):
    """→ калиброванная P(анемия), уровень риска 0/1/2 (до согласования с Hb), прогноз Hb (г/л), доля признаков
    вне обучающего диапазона. Модель без 'cal' (обучена до калибровки) — P без калибровки, как раньше."""
    X = np.atleast_2d(np.asarray(X, float))
    p = calibrate(B.get('cal'), _p_anemia(B['cls'], B['prior'], X))
    return p, risk_level(p, B['t_lo'], B['t_hi']), B['hb'].predict(X), ood_share(X, B['ood_lo'], B['ood_hi'])


# ------------------------------------------------------------------ проверка на фолдах манифеста
def cross_validate(F, feats, dataset, S=None, seed=0, train_by_fold=None):
    """OOF по фолдам манифеста: модель и все её пороги обучаются на 4 фолдах, применяются к пятому.
    S — признаки тех же снимков при 8 стресс-условиях света (group, cond): риск по ним считает модель того фолда,
    где пациент был тестовым. train_by_fold — {фолд f: обучающие строки}, если признаки обучающих снимков зависят
    от тестового фолда (маски сегментатора, не видевшего f); иначе обучение — строки F остальных фолдов.
    → (OOF-таблица исходных снимков, стресс-таблица или None)."""
    O = F[F.aug == 0].reset_index(drop=True)
    cols = {k: np.full(len(O), np.nan) for k in ('p', 'p_raw', 'level', 'level_p', 'raised', 'hb_pred', 'ood',
                                                 'hb_ok', 'thr_sens', 'hb_hw')}
    who = O.who_normal.values if 'who_normal' in O else None
    stress = []
    for f in sorted(O.fold.unique()):
        te = np.flatnonzero(O.fold.values == f)
        T = F[F.fold != f] if train_by_fold is None else train_by_fold[f]
        assert not set(T.group) & set(O.group.values[te]), 'пациент теста в обучении'
        B = fit_core(T, feats, dataset, seed)
        p, lv, hbp, ood = apply_core(B, O[feats].values[te])
        up = np.zeros(len(te), bool)
        lv_p = lv
        if who is not None:
            lv, up = reconcile(lv, hbp, B['hb_halfwidth'], B['hb_ok'], who[te])
        p_raw = _p_anemia(B['cls'], B['prior'], O[feats].values[te])
        for k, v in (('p', p), ('p_raw', p_raw), ('level', lv), ('level_p', lv_p), ('raised', up),
                     ('hb_pred', hbp), ('ood', ood)):
            cols[k][te] = v
        cols['hb_ok'][te], cols['thr_sens'][te], cols['hb_hw'][te] = B['hb_ok'], B['thr_sens'], B['hb_halfwidth']
        if S is not None and len(S):
            St = S[S.group.isin(set(O.group.values[te]))]
            if len(St):
                ps, ls, hbs, _ = apply_core(B, St[feats].values)
                if 'who_normal' in St:
                    ls, _ = reconcile(ls, hbs, B['hb_halfwidth'], B['hb_ok'], St.who_normal.values)
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
    r.update(M.binary_report(flag, anem, g))       # бинарный ответ — тот же порог thr_sens (вложенная CV на train)
    r.update(calibration_metrics(p, anem, O.p_raw.values if 'p_raw' in O else None))
    lvp = O.level_p.values.astype(int) if 'level_p' in O else lv
    for name, L in (('', lv), ('_p', lvp)):        # итог и «только по P» (до согласования с Hb)
        r['zsens' + name], r['zsens' + name + '_lo'], r['zsens' + name + '_hi'] = M.wilson(
            int((anem & (L >= 1)).sum()), int(anem.sum()))
        r['zspec' + name], r['zspec' + name + '_lo'], r['zspec' + name + '_hi'] = M.wilson(
            int((~anem & (L == 0)).sum()), int((~anem).sum()))
    up = O.raised.values.astype(bool) if 'raised' in O else np.zeros(len(O), bool)
    r['n_raised'], r['raised_anemic'] = int(up.sum()), int((up & anem).sum())
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


# ------------------------------------------------------------------ калибровка: метрики и график надёжности
def reliability(p, y, bins=10):
    """Таблица надёжности: равные по ширине корзины P → n, средняя P, доля больных (Уилсон 95%)."""
    p, y = np.asarray(p, float), np.asarray(y).astype(bool)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        s = idx == b
        if s.any():
            frac, lo, hi = M.wilson(int(y[s].sum()), int(s.sum()))
            rows.append({'bin': f'{b / bins:.1f}–{(b + 1) / bins:.1f}', 'n': int(s.sum()), 'p_mean': float(p[s].mean()),
                         'frac_anemic': frac, 'frac_lo': lo, 'frac_hi': hi})
    return pd.DataFrame(rows)


def ece(p, y, bins=10):
    """Ожидаемая ошибка калибровки: Σ (n_b / n) · |средняя P − доля больных| по равным корзинам."""
    t = reliability(p, y, bins)
    return float((t.n * (t.p_mean - t.frac_anemic).abs()).sum() / t.n.sum()) if len(t) else float('nan')


def calibration_metrics(p, y, p_raw=None):
    """Brier и ECE калиброванной P (и сырой, если дана); brier_ref — константа = доля больных (ниже — лучше)."""
    y = np.asarray(y).astype(float)
    r = {'brier': float(np.mean((np.asarray(p, float) - y) ** 2)), 'ece': ece(p, y),
         'brier_ref': float(np.mean((y.mean() - y) ** 2))}
    if p_raw is not None:
        r['brier_raw'], r['ece_raw'] = float(np.mean((np.asarray(p_raw, float) - y) ** 2)), ece(p_raw, y)
    return r
