"""Единственный источник цифр точности. Каждая строка результата дописывается в results/metrics_history.csv
и в <data>/results/metrics_history.csv (вне git: не теряется при обновлении репозитория в Colab).

    python scripts/evaluate.py --data data --tag baseline_v11            # всё, что есть в манифесте
    python scripts/evaluate.py --task hb --dataset eyes_defy --model ridge

Задачи:
  seg  — baseline-сегментация v11 (Eyes-defy): IoU/Dice с масками врачей на невиданных пациентах.
  v12  — модель демо-блокнота v12 (anemia/v12_model.py): 3 класса ВОЗ по конъюнктиве / склере / обеим.
         Сбалансированная точность (это «61%» блокнота), AUC, чувствительность/специфичность по argmax (v12.0)
         и по порогу P(анемия), подобранному внутри обучающих фолдов (v12.1); свет — доля пациентов,
         у которых ответ «анемия да/нет» меняется в 8 стресс-условиях.
  hb   — Hb по цвету конъюнктивы: MAE, Bland–Altman, AUC, чувствительность/специфичность при пороге,
         подобранном внутри обучающих фолдов, стресс-тест «другой свет».
         Маска: Eyes-defy — маска врача (потолок) и маска baseline-сегментации; без масок врача — весь кадр.
  core — ядро HemaVision (anemia/core.py, то же, что обучает scripts/train_core.py): риск низкий/средний/высокий
         по двум порогам P(анемия), Hb (экспериментально), OOD; все пороги — вложенной CV на обучающих фолдах.
         Только явно: --task core (в --task all не входит). Признаки — те же, что у v12 (общий кэш).
  country — внешняя проверка ядра внутри Eyes-defy: обучение на всех пациентах Индии → тест на всех пациентах
         Италии и наоборот; сегментатор, модель, пороги и OOD обучаются только на стране обучения. Плюс базовая
         модель «всегда частый класс страны обучения». Только явно: --task country. Ограничения — CLAUDE.md, «Как устроена оценка».
Для каждого бинарного ответа «анемия» (положительный класс — анемия по ВОЗ): матрица ошибок TP/FP/FN/TN,
accuracy, precision (PPV), recall (= чувствительность), специфичность, NPV (ДИ Уилсона), F1 и сбалансированная
точность двух классов (бутстрэп-ДИ по пациентам); не определённая метрика (нулевой знаменатель) — «—», не 0.
Все метрики — по фиксированным фолдам манифеста (по пациенту). OOF-предсказания → <data>/results/.

Каждый запуск получает ID и свою папку <data>/results/runs/<run_id>/ (run.json: коммит, параметры, время,
тайминги; metrics.csv; OOF; лог), которая никогда не перезаписывается.
Признаки снимков кэшируются в <data>/cache/features/ (anemia/cache.py: ключ = содержимое файлов + код +
параметры; меток в кэше нет). --no-cache — без кэша, --refresh-cache — пересчитать и перезаписать.
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
import time

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anemia import core as CORE  # noqa: E402
from anemia import metrics as M  # noqa: E402
from anemia import predict as PR  # noqa: E402
from anemia import segment_baseline as SB  # noqa: E402
from anemia import v12_model as V12  # noqa: E402
from anemia import runs as RUNS  # noqa: E402
from anemia import external as EXT  # noqa: E402
from anemia.data import BAD_DOCTOR_MASKS  # noqa: E402
from anemia.augment import STRESS_CONDITIONS, apply_illuminant  # noqa: E402
from anemia.cache import TIMING_KEYS, FeatureCache, empty_timings, file_sha1, source_version  # noqa: E402
from anemia.features import N_RELIGHT, WORK_SIZE, feature_rows, load_item, row_meta  # noqa: E402
from anemia.hb_models import cross_validate  # noqa: E402
from anemia.io import load_mask, load_rgb, resize_to  # noqa: E402
from anemia.labs import is_pregnant  # noqa: E402
from anemia.regions import detect_regions  # noqa: E402

HISTORY = os.path.join(ROOT, 'results', 'metrics_history.csv')
HIST_COLS = ['timestamp', 'commit', 'tag', 'task', 'dataset', 'method', 'n',
             'mae', 'mae_lo', 'mae_hi', 'bias', 'loa_lo', 'loa_hi', 'auc', 'auc_lo', 'auc_hi',
             'target_sens', 'sens', 'sens_lo', 'sens_hi', 'spec', 'spec_lo', 'spec_hi',
             'n_hb100', 'sens_hb100', 'light_med', 'light_p90',
             'iou_mean', 'iou_lo', 'iou_hi', 'iou_median', 'dice_mean', 'found',
             'bal_acc', 'bal_acc_lo', 'bal_acc_hi', 'auc_light', 'light_flip',
             'tp', 'fp', 'fn', 'tn', 'acc', 'acc_lo', 'acc_hi', 'prec', 'prec_lo', 'prec_hi', 'rec', 'npv', 'npv_lo',
             'npv_hi', 'f1', 'f1_lo', 'f1_hi', 'bacc2', 'bacc2_lo', 'bacc2_hi', 'n_accepted', 'n_ood']


NO_CACHE = FeatureCache(None, 'off')


class Tee:
    def __init__(self, *paths):
        self.fs, self.out = [open(p, 'w') for p in paths], sys.stdout

    def write(self, s):
        self.out.write(s)
        for f in self.fs:
            f.write(s)

    def flush(self):
        self.out.flush()
        for f in self.fs:
            f.flush()


class StageTimer:
    """Время по этапам: чтение снимков, признаки, кэш — суммы по рабочим процессам; «стена» — по часам."""

    def __init__(self):
        self.rows = []

    def add(self, stage, timings, wall):
        tot = {k: float(sum(t[k] for t in timings)) for k in TIMING_KEYS}
        self.rows.append({'stage': stage, 'n': len(timings), **tot, 'wall': wall})

    def table(self):
        head = ['Этап', 'снимков', 'из кэша', 'чтение снимков, с', 'признаки, с', 'ключ кэша (sha1 файлов), с',
                'кэш чтение / запись, с', 'стена, с']
        out = ['| ' + ' | '.join(head) + ' |', '|' + '---|' * len(head)]
        for r in self.rows:
            out.append(f"| {r['stage']} | {r['n']} | {r['hit']:.0f} | {r['read']:.1f} | {r['features']:.1f} | "
                       f"{r['key']:.1f} | {r['cache_read']:.1f} / {r['cache_write']:.1f} | {r['wall']:.1f} |")
        return '\n'.join(out)


GENERATED = ('results/metrics_history.csv',)   # файлы, которые дописывает сам evaluate.py: не правка кода


def git_state(root=ROOT):
    """→ {'commit': '<sha>[-dirty]', 'dirty_code': [...], 'dirty_results': [...]}.
    «-dirty» — только если изменены отслеживаемые файлы, кроме GENERATED (их дописывают прошлые запуски)."""
    try:
        h = subprocess.run(['git', '-C', root, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                           check=True).stdout.strip()
        st = subprocess.run(['git', '-C', root, 'status', '--porcelain', '--untracked-files=no'],
                            capture_output=True, text=True, check=True).stdout
    except Exception:
        return {'commit': 'unknown', 'dirty_code': [], 'dirty_results': []}
    files = [line[3:].split(' -> ')[-1] for line in st.splitlines() if line.strip()]
    code = [f for f in files if f not in GENERATED]
    return {'commit': h + ('-dirty' if code else ''), 'dirty_code': code,
            'dirty_results': [f for f in files if f in GENERATED]}


def git_commit():
    return git_state()['commit']


# ------------------------------------------------------------------ сегментация
def _seg_prep(root, row):
    t0 = time.perf_counter()
    rgb, doc = load_item(root, row, 'doctor')
    t1 = time.perf_counter()
    F = SB.pixel_features(rgb)
    return (rgb, doc, F), {**empty_timings(), 'read': t1 - t0, 'features': time.perf_counter() - t1, 'miss': 1.0}


def run_seg(root, df, jobs, timer=None):
    """OOF-маски baseline-сегментации по фолдам манифеста + метрики.
    Попиксельные признаки (~7 МБ на снимок) не кэшируются: снимки здесь читаются каждый запуск."""
    t0 = time.perf_counter()
    out = Parallel(n_jobs=jobs)(delayed(_seg_prep)(root, r) for r in df.to_dict('records'))
    items = [it for it, _ in out]
    if timer is not None:
        timer.add('сегментация v11: чтение + признаки пикселей', [t for _, t in out], time.perf_counter() - t0)
    pred = [None] * len(df)
    folds = df.fold.values
    good = ~df.patient_id.isin(BAD_DOCTOR_MASKS).values      # испорченные маски: не учим и не меряем по ним
    for f in sorted(set(folds)):
        tr = [i for i in range(len(df)) if folds[i] != f and good[i]]
        model = SB.train([items[i][2] for i in tr], [items[i][1] for i in tr])
        for i in np.flatnonzero(folds == f):
            pred[i] = SB.predict(model, items[i][2])
    ious = np.array([M.iou(p, it[1]) for p, it in zip(pred, items)])[good]
    dices = np.array([M.dice(p, it[1]) for p, it in zip(pred, items)])[good]
    lo, hi = M.bootstrap_ci(np.mean, df.group.values[good], 1000, 0, ious)
    res = {'n': int(good.sum()), 'iou_mean': ious.mean(), 'iou_lo': lo, 'iou_hi': hi,
           'iou_median': float(np.median(ious)), 'dice_mean': dices.mean(),
           'found': float(np.mean([p.any() for p, g in zip(pred, good) if g]))}
    return res, pred, items


# ------------------------------------------------------------------ Hb
def _hb_compute(root, row, mask_source, mask, seed):
    """Признаки без меток: исходный снимок, копии при другом свете, стресс-условия."""
    t0 = time.perf_counter()
    if mask is None:
        rgb, mask = load_item(root, row, mask_source)
    else:
        rgb, _ = load_item(root, row, 'full')
    t1 = time.perf_counter()
    out = feature_rows(rgb, mask, {}, seed)
    return out, {'read': t1 - t0, 'features': time.perf_counter() - t1}


def _feat_job(root, row, mask_source, mask, seed, cache=NO_CACHE):
    files = {'image': os.path.join(root, row['image'])}
    if mask is None and mask_source == 'doctor':
        files['mask'] = os.path.join(root, row['mask_conj'])
    params = {'mask_source': mask_source if mask is None else 'array', 'seed': seed, 'work_size': WORK_SIZE,
              'n_relight': N_RELIGHT, 'stress': STRESS_CONDITIONS}
    (rows, srows), t = cache.fetch('hb', files, {} if mask is None else {'mask': mask}, params,
                                   _hb_compute, root, row, mask_source, mask, seed)
    meta = row_meta(row)                                  # метки — из текущего манифеста, не из кэша
    return ([{**meta, **r} for r in rows], [{**meta, **r} for r in srows]), t


def hb_metrics(O, S, target):
    y, pred, true = O.anemic.values.astype(bool), O.pred.values, O.hb_g_l.values
    g = O.group.values
    r = {'n': len(O), 'mae': M.mae(pred, true), 'target_sens': target, **M.bland_altman(pred, true)}
    r['mae_lo'], r['mae_hi'] = M.bootstrap_ci(M.mae, g, 1000, 0, pred, true)
    if 0 < y.sum() < len(y):
        r['auc'] = M.auc(O.score.values, y)
        r['auc_lo'], r['auc_hi'] = M.bootstrap_ci(lambda s, t: M.auc(s, t), g, 1000, 0, O.score.values, y)
    flag = O.flag.values
    r['sens'], r['sens_lo'], r['sens_hi'] = M.wilson(int((flag & y).sum()), int(y.sum()))
    r['spec'], r['spec_lo'], r['spec_hi'] = M.wilson(int((~flag & ~y).sum()), int((~y).sum()))
    r.update(M.binary_report(flag, y, g))
    low = true < 100
    r['n_hb100'] = int(low.sum())
    r['sens_hb100'] = float(flag[low].mean()) if low.any() else float('nan')
    if S is not None and len(S):
        spread = S.groupby('group').pred.agg(lambda p: (p.max() - p.min()) / 2)
        r['light_med'], r['light_p90'] = float(spread.median()), float(spread.quantile(0.9))
    return r


def run_hb(root, df, mask_source, masks, model_kind, target, jobs, cache=NO_CACHE, timer=None, stage='Hb'):
    t0 = time.perf_counter()
    res = Parallel(n_jobs=jobs)(delayed(_feat_job)(root, r, mask_source, None if masks is None else masks[i], i,
                                                   cache)
                                for i, r in enumerate(df.to_dict('records')))
    out = [o for o, _ in res]
    if timer is not None:
        timer.add(stage, [t for _, t in res], time.perf_counter() - t0)
    F = pd.DataFrame([x for rows, _ in out for x in rows])
    S = pd.DataFrame([x for _, srows in out for x in srows])
    n_skip = len(df) - F[F.aug == 0].group.nunique() if len(F) else len(df)
    feats = [c for c in F.columns if c.startswith('c_')]
    if df.sex.notna().any():
        feats.append('male')
    O, SP = cross_validate(F, feats, model_kind, target, stress=S if len(S) else None)
    return hb_metrics(O, SP, target), O, n_skip


# ------------------------------------------------------------------ v12
def _v12_compute(path, seed):
    """Признаки v12 одного снимка без меток: исходный, N_RELIGHT обучающих копий при другом свете, стресс-условия."""
    t0 = time.perf_counter()
    img = load_rgb(path, V12.DETECT_SIDE)
    t1 = time.perf_counter()
    cv2.setRNGSeed(0)                                     # GrabCut в поиске тканей случаен: без сида маски плывут
    crop, conj, sclera = V12.detect(img)
    rows = [{'aug': 0, **V12.eye_features(crop, conj, sclera)}]
    rng = np.random.default_rng(seed)
    for a in range(1, V12.N_RELIGHT + 1):
        rows.append({'aug': a, **V12.eye_features(V12.relight(crop, rng), conj, sclera)})
    srows = [{'cond': f'{T}K×{e}', **V12.eye_features(apply_illuminant(crop, T, e), conj, sclera)}
             for T, e in STRESS_CONDITIONS]
    return (rows, srows), {'read': t1 - t0, 'features': time.perf_counter() - t1}


def _v12_job(root, row, seed, cache=NO_CACHE):
    path = os.path.join(root, row['image'])
    params = {'seed': seed, 'detect_side': V12.DETECT_SIDE, 'crop_side': V12.CROP_SIDE,
              'n_relight': V12.N_RELIGHT, 'stress': STRESS_CONDITIONS}
    (rows, srows), t = cache.fetch('v12', {'image': path}, {}, params, _v12_compute, path, seed)
    return _with_meta(row, rows, srows), t


def _with_meta(row, rows, srows):
    age = row['age'] if row['age'] == row['age'] else 30.0      # метки — из текущего манифеста, не из кэша
    sex = row['sex'] if isinstance(row['sex'], str) else None
    meta = {'group': row['group'], 'fold': int(row['fold']), 'anemic': bool(row['anemic']),
            'hb_g_l': float(row['hb_g_l']),
            'y': V12.hb_class(row['hb_g_l'] / 10, *V12.who_thresholds(sex, age, is_pregnant(row['pregnant']))),
            'male': 1.0 if sex == 'M' else 0.0 if sex == 'F' else 0.5}
    return [{**meta, **r} for r in rows], [{**meta, **r} for r in srows]


def v12_decide(P, thr=None):
    """Класс: argmax (v12.0) или порог P(анемия) (v12.1: выше порога — не ниже «лёгкой», ниже — «норма»)."""
    k = P.argmax(1)
    return k if thr is None else np.where(1 - P[:, 0] >= thr, np.maximum(k, 1), 0)


def v12_cv(F, S, feats, target, inner_folds=4):
    """OOF по фолдам манифеста. Порог P(анемия) — вложенной CV только на обучающих фолдах."""
    O = F[F.aug == 0].reset_index(drop=True)
    P = np.zeros((len(O), 3))
    thr = np.full(len(O), np.nan)
    stress = []
    for f in sorted(O.fold.unique()):
        tr_all, te = F[F.fold != f], np.flatnonzero(O.fold.values == f)
        tr0 = O[O.fold != f].reset_index(drop=True)
        prior = V12.class_prior(tr0.y.values)
        m = V12.fit(tr_all[feats].values, tr_all.y.values)
        P[te] = V12.predict(m, O[feats].values[te], prior)
        St = S[S.group.isin(O.group.values[te])]
        inner = np.zeros((len(tr0), 3))
        for itr, ite in GroupKFold(min(inner_folds, tr0.group.nunique())).split(tr0, groups=tr0.group):
            T = tr_all[tr_all.group.isin(set(tr0.group.iloc[itr]))]
            inner[ite] = V12.predict(V12.fit(T[feats].values, T.y.values), tr0[feats].values[ite],
                                     V12.class_prior(tr0.y.values[itr]))
        thr[te] = M.threshold_for_sensitivity(1 - inner[:, 0], tr0.anemic.values, target)
        if len(St):
            stress.append(St.assign(P=list(V12.predict(m, St[feats].values, prior)), thr=thr[te][0]))
    O['P'], O['thr'] = list(P), thr
    return O, (pd.concat(stress) if stress else None)


def v12_metrics(O, S, use_thr, target):
    P, y, anem, g = np.stack(O.P.values), O.y.values, O.anemic.values.astype(bool), O.group.values
    thr = O.thr.values if use_thr else None
    k = np.array([v12_decide(P[i:i + 1], None if thr is None else thr[i])[0] for i in range(len(P))])
    flag = k > 0
    r = {'n': len(O), 'bal_acc': float(balanced_accuracy_score(y, k)), 'target_sens': target if use_thr else None}
    r['bal_acc_lo'], r['bal_acc_hi'] = M.bootstrap_ci(balanced_accuracy_score, g, 1000, 0, y, k)
    if 0 < anem.sum() < len(anem):
        r['auc'] = M.auc(1 - P[:, 0], anem)
        r['auc_lo'], r['auc_hi'] = M.bootstrap_ci(M.auc, g, 1000, 0, 1 - P[:, 0], anem)
    r['sens'], r['sens_lo'], r['sens_hi'] = M.wilson(int((flag & anem).sum()), int(anem.sum()))
    r['spec'], r['spec_lo'], r['spec_hi'] = M.wilson(int((~flag & ~anem).sum()), int((~anem).sum()))
    r.update(M.binary_report(flag, anem, g))
    if S is not None and len(S):
        PS = np.stack(S.P.values)
        fl = (1 - PS[:, 0] >= S.thr.values) if use_thr else (PS.argmax(1) > 0)
        per_cond = [M.auc(1 - PS[(S.cond == c).values, 0], S.anemic.values[(S.cond == c).values].astype(bool))
                    for c in S.cond.unique()]
        r['auc_light'] = float(np.mean(per_cond))
        r['light_flip'] = float(pd.Series(fl, index=S.group.values).groupby(level=0).nunique().gt(1).mean())
    return r


def run_v12(root, df, target, jobs, cache=NO_CACHE, timer=None):
    t0 = time.perf_counter()
    res = Parallel(n_jobs=jobs)(delayed(_v12_job)(root, r, i, cache) for i, r in enumerate(df.to_dict('records')))
    out = [o for o, _ in res]
    if timer is not None:
        timer.add(f'v12 {df.dataset.iloc[0]}: чтение + поиск тканей + признаки', [t for _, t in res],
                  time.perf_counter() - t0)
    F = pd.DataFrame([x for rows, _ in out for x in rows])
    S = pd.DataFrame([x for _, srows in out for x in srows])
    sets = V12.feature_sets(F.columns, df.sex.notna().any())
    res = []
    for mode, feats in sets.items():
        if len(feats) < 3:
            continue
        ok = set(F[F.aug == 0].dropna(subset=feats).group)            # пациенты, у которых ткань нашлась
        Fm, Sm = F[F.group.isin(ok)].dropna(subset=feats), S[S.group.isin(ok)].dropna(subset=feats)
        if Fm.group.nunique() < 40:
            continue
        O, SP = v12_cv(Fm, Sm, feats, target)
        for use_thr in (False, True):
            r = v12_metrics(O, SP, use_thr, target)
            res.append((mode, 'порог 0.80' if use_thr else 'argmax', len(df) - O.group.nunique(), r, O))
    return res


# ------------------------------------------------------------------ ядро HemaVision
def v12_features(root, df, jobs, cache=NO_CACHE, timer=None, stage='признаки v12'):
    """Признаки v12 всех снимков датасета (общий кэш с --task v12) → (F, S)."""
    t0 = time.perf_counter()
    res = Parallel(n_jobs=jobs)(delayed(_v12_job)(root, r, i, cache) for i, r in enumerate(df.to_dict('records')))
    if timer is not None:
        timer.add(f'{stage} {df.dataset.iloc[0]}: чтение + поиск тканей + признаки', [t for _, t in res],
                  time.perf_counter() - t0)
    F = pd.DataFrame([x for (rows, _), _ in res for x in rows])
    S = pd.DataFrame([x for (_, srows), _ in res for x in srows])
    return F, S


# ------------------------------------------------------------------ ядро: конъюнктива обученным сегментатором
def _segpix_compute(root, row, seed):
    t0 = time.perf_counter()
    rgb, doc = load_item(root, row, 'doctor')
    t1 = time.perf_counter()
    return SB.image_samples(rgb, doc, seed), {'read': t1 - t0, 'features': time.perf_counter() - t1}


def _segpix_job(root, row, seed, cache=NO_CACHE):
    files = {'image': os.path.join(root, row['image']), 'mask': os.path.join(root, row['mask_conj'])}
    params = {'seed': seed, 'seg_size': SB.SEG_SIZE, 'work_size': WORK_SIZE, 'pix_per_image': SB.PIX_PER_IMAGE}
    return cache.fetch('segpix', files, {}, params, _segpix_compute, root, row, seed)


def _segpix_ext_compute(root, row, seed):
    t0 = time.perf_counter()
    rgb = load_rgb(os.path.join(root, row['file']), WORK_SIZE)
    gt = resize_to(load_mask(os.path.join(root, row['mask'])), rgb.shape[:2], nearest=True)
    t1 = time.perf_counter()
    return SB.image_samples(rgb, gt, seed), {'read': t1 - t0, 'features': time.perf_counter() - t1}


def _segpix_ext_job(root, row, seed, cache=NO_CACHE):
    files = {'image': os.path.join(root, row['file']), 'mask': os.path.join(root, row['mask'])}
    params = {'seed': seed, 'work_size': WORK_SIZE, 'pix_per_image': SB.PIX_PER_IMAGE}
    return cache.fetch('segpix_ext', files, {}, params, _segpix_ext_compute, root, row, seed)[0]


def external_seg_samples(root, jobs, cache=NO_CACHE, idx=None):
    """Пиксели внешних телефонных фото с масками конъюнктивы (anemia.external.seg_index). Пациентов Eyes-defy там
    нет (сверка sha1 и миниатюр), Hb и меток риска нет — добавляются в обучение сегментатора (не модели риска).
    Их IoU вне обучения — только --task seg_ext (фолды по группам). Нет индекса — пустой список."""
    idx = EXT.seg_index(root) if idx is None else idx
    return Parallel(n_jobs=jobs)(delayed(_segpix_ext_job)(root, r, 100000 + i, cache)
                                 for i, r in enumerate(idx.to_dict('records')))


def conj_segmenters(root, df, jobs, cache=NO_CACHE, timer=None, nested=False, full=True, external=True):
    """Сегментаторы конъюнктивы по маскам врачей → {набор фолдов, которых модель НЕ видела: модель}.
        (g,)   — маски снимков фолда g (OOF): тестовые снимки evaluate и обучающие признаки train_core;
        (f, g) — nested=True, только evaluate: маски обучающих снимков фолда g при оценке внешнего фолда f. Так маски
                 врачей тестового фолда f не влияют ни на одну маску, по которой учится модель риска фолда f;
        ()     — full=True: модель на всех снимках, только для новых фото (файл модели, predict).
    external=True — в обучение каждой модели добавляются телефонные фото с масками (external_seg_samples).
    Hb и метки риска сегментатор не видит."""
    assert WORK_SIZE == SB.SEG_SIZE, 'сегментатор обучается на load_item (WORK_SIZE), применяется на SEG_SIZE'
    t0 = time.perf_counter()
    res = Parallel(n_jobs=jobs)(delayed(_segpix_job)(root, r, i, cache) for i, r in enumerate(df.to_dict('records')))
    if timer is not None:
        timer.add(f'сегментатор конъюнктивы {df.dataset.iloc[0]}: пиксели масок врачей', [t for _, t in res],
                  time.perf_counter() - t0)
    samples, fold_of = [x for x, _ in res], df.fold.astype(int).values
    folds = sorted(set(fold_of))
    # испорченные маски (BAD_DOCTOR_MASKS) в обучение сегментатора не идут: как будто снимок в фолде -1
    fold_of = np.where(df.patient_id.isin(BAD_DOCTOR_MASKS).values, -1, fold_of)
    keys = ([()] if full else []) + [(g,) for g in folds] + (
        [(f, g) for f in folds for g in folds if g != f] if nested else [])
    t0 = time.perf_counter()
    ext = external_seg_samples(root, jobs, cache) if external else []
    if ext:
        print(f'сегментатор конъюнктивы: + {len(ext)} телефонных фото с масками (external/)', flush=True)
    models = Parallel(n_jobs=jobs)(delayed(SB.fit_samples)([x for x, g in zip(samples, fold_of) if g not in k and g >= 0]
                                                           + ext) for k in keys)
    if timer is not None:
        timer.add(f'сегментатор конъюнктивы {df.dataset.iloc[0]}: обучение {len(keys)} моделей (+{len(ext)} внешних '
                  'телефонных фото)', [], time.perf_counter() - t0)
    return dict(zip(keys, models))


def _seg_test_item(root, row, kind):
    if kind == 'ed':
        return load_item(root, row, 'doctor')
    rgb = load_rgb(os.path.join(root, row['file']), WORK_SIZE)
    return rgb, resize_to(load_mask(os.path.join(root, row['mask'])), rgb.shape[:2], nearest=True)


def _seg_scores(model, items):
    """(IoU, Dice, нашёл) по снимкам; пустая разметка → None (IoU пусто/пусто = 1 завысил бы среднее)."""
    out = []
    for rgb, gt in items:
        if not gt.any():
            out.append(None)
            continue
        p = SB.conj_mask(model, rgb)
        out.append((M.iou(p, gt), M.dice(p, gt), bool(p.any())))
    return out


def run_seg_external(root, df, jobs, cache=NO_CACHE, timer=None, n_folds=5):
    """Сегментатор ядра (SB.conj_mask, как в анализе фото) без и с телефонными фото external/ — IoU вне обучения:
    Eyes-defy по фолдам манифеста, телефонные фото по GroupKFold их групп (почти одинаковые кадры в одном фолде).
    Модель фолда f не видит ни снимков Eyes-defy фолда f, ни телефонных фото фолда f. → строки для истории."""
    idx = EXT.seg_index(root)
    if not len(idx):
        print('нет <data>/external/index.csv с масками (python scripts/fetch_data.py --external phone_conj_masks) — '
              'пропуск seg_ext')
        return []
    t0 = time.perf_counter()
    ed_s = Parallel(n_jobs=jobs)(delayed(_segpix_job)(root, r, i, cache) for i, r in enumerate(df.to_dict('records')))
    ex_s = external_seg_samples(root, jobs, cache, idx)
    ed_items = Parallel(n_jobs=jobs)(delayed(_seg_test_item)(root, r, 'ed') for r in df.to_dict('records'))
    ex_items = Parallel(n_jobs=jobs)(delayed(_seg_test_item)(root, r, 'ext') for r in idx.to_dict('records'))
    if timer is not None:
        timer.add('seg_ext: пиксели масок и тестовые снимки', [t for _, t in ed_s], time.perf_counter() - t0)
    ed_s = [x for x, _ in ed_s]
    good = ~df.patient_id.isin(BAD_DOCTOR_MASKS).values
    ed_fold = df.fold.astype(int).values
    ex_fold = np.zeros(len(idx), int)
    for f, (_, te) in enumerate(GroupKFold(min(n_folds, idx.group.nunique())).split(idx, groups=idx.group)):
        ex_fold[te] = f
    folds = sorted(set(ed_fold))
    jobs_list = []
    for f in folds:
        tr_ed = [x for x, g, ok in zip(ed_s, ed_fold, good) if g != f and ok]
        tr_ex = [x for x, g in zip(ex_s, ex_fold) if g != f]
        for arm, tr in (('Eyes-defy', tr_ed), ('Eyes-defy + телефонные', tr_ed + tr_ex)):
            jobs_list.append((f, arm, tr))
    models = Parallel(n_jobs=jobs)(delayed(SB.fit_samples)(tr) for _, _, tr in jobs_list)
    scores = {}
    for (f, arm, _), m in zip(jobs_list, models):
        te_ed = [it for it, g, ok in zip(ed_items, ed_fold, good) if g == f and ok]
        te_ex = [it for it, g in zip(ex_items, ex_fold) if g == f]
        scores.setdefault((arm, 'eyes_defy'), []).extend(
            zip(df.group.values[(ed_fold == f) & good], _seg_scores(m, te_ed)))
        scores.setdefault((arm, 'phone_conj_masks'), []).extend(
            zip(idx.group.values[ex_fold == f], _seg_scores(m, te_ex)))
    rows = []
    for (arm, test), v in scores.items():
        v = [(g, sc) for g, sc in v if sc is not None]
        groups = np.array([g for g, _ in v])
        iou = np.array([s[0] for _, s in v])
        lo, hi = M.bootstrap_ci(np.mean, groups, 1000, 0, iou)
        rows.append({'task': 'seg_ext', 'dataset': test, 'method': f'сегментатор ядра · обучение: {arm}',
                     'n': len(v), 'iou_mean': float(iou.mean()), 'iou_lo': lo, 'iou_hi': hi,
                     'iou_median': float(np.median(iou)), 'dice_mean': float(np.mean([s[1] for _, s in v])),
                     'found': float(np.mean([s[2] for _, s in v]))})
    return rows


DET_KEYS = ('crop', 'conj', 'sclera', 'iris', 'use', 'kind', 'notes', 'src_side', 'score')


def _det_compute(path):
    """Шаг 1 анализа фото (predict._detect): весь кадр DETECT_SIDE, поиск тканей v12 с тем же сидом GrabCut."""
    t0 = time.perf_counter()
    img = load_rgb(path, V12.DETECT_SIDE)
    t1 = time.perf_counter()
    cv2.setRNGSeed(0)
    det = detect_regions(img, 'both')
    return {k: det[k] for k in DET_KEYS}, {'read': t1 - t0, 'features': time.perf_counter() - t1}


def eye_route(det, seg):
    """Снимок глаза → (ткани после тех же отказов, что в анализе фото; модель взрослых или None; причина отказа).
    Шаг 1 _detect: несогласованные ткани → отказ; конъюнктива — сегментатор seg (learned_conj, с его отказами);
    затем качество тканей и выбор модели (tissue_route). Остальные шаги _detect (лицо, мелкая радужка) на
    Eyes-defy меняют маршрут 1 снимка из 216 (сверка 2026-10-18) и здесь не повторяются."""
    why = PR._inconsistent(det)
    det = PR.learned_conj(PR._rejected(det, why) if why else det, seg)
    key, why = PR.tissue_route(det)
    return det, key, why


def _seg_features(crop, conj, sclera, seed, stress):
    t0 = time.perf_counter()
    rows = [{'aug': 0, **V12.eye_features(crop, conj, sclera)}]
    rng = np.random.default_rng(seed)
    for a in range(1, V12.N_RELIGHT + 1):
        rows.append({'aug': a, **V12.eye_features(V12.relight(crop, rng), conj, sclera)})
    srows = [{'cond': f'{T}K×{e}', **V12.eye_features(apply_illuminant(crop, T, e), conj, sclera)}
             for T, e in (STRESS_CONDITIONS if stress else [])]
    return (rows, srows), {'read': 0.0, 'features': time.perf_counter() - t0}


def _core_job(root, row, seed, seg, stress, cache=NO_CACHE):
    """Один снимок глаза: маршрут анализа фото + признаки тканей той модели, которой он будет оценён."""
    path = os.path.join(root, row['image'])
    det, t = cache.fetch('v12det', {'image': path}, {}, {'detect_side': V12.DETECT_SIDE}, _det_compute, path)
    det, key, why = eye_route(det, seg)
    if key is None:
        return {'key': None, 'why': why, 'rows': [], 'srows': []}, t
    tissues = CORE.SPECS[key]['tissues']              # как analyze_photo: только пригодные ткани
    conj = det['conj'] if 'conj' in tissues else None
    sclera = det['sclera'] if 'sclera' in tissues else None
    arrays = {'crop': det['crop'], **({} if conj is None else {'conj': conj}),
              **({} if sclera is None else {'sclera': sclera})}
    params = {'seed': seed, 'n_relight': V12.N_RELIGHT, 'stress': STRESS_CONDITIONS if stress else None}
    (rows, srows), t2 = cache.fetch('v12seg', {}, arrays, params, _seg_features, det['crop'], conj, sclera, seed,
                                    stress)
    rows, srows = _with_meta(row, rows, srows)
    return {'key': key, 'why': '', 'rows': rows, 'srows': srows}, {k: t[k] + t2[k] for k in t}


def _core_batch(root, items, seg, stress, cache):
    return [(i, role, *_core_job(root, row, i, seg, stress, cache)) for i, role, row in items]


def core_jobs(root, df, jobs, segs, cache=NO_CACHE, timer=None, nested=False):
    """Маршрут и признаки каждого снимка глаза → список {i, role, key, why, rows, srows}.
    role 'test' — снимок фолда g с маской сегментатора без g (+ стресс-условия света);
    role f (nested) — тот же снимок как обучающий при внешнем фолде f, маска сегментатора без f и g."""
    recs = list(df.to_dict('records'))
    batches = {}
    for i, row in enumerate(recs):
        g = int(row['fold'])
        batches.setdefault((g,), []).append((i, 'test', row))
        for f in sorted(set(df.fold.astype(int))) if nested else []:
            if f != g:
                batches.setdefault((f, g), []).append((i, f, row))
    t0 = time.perf_counter()
    out = Parallel(n_jobs=jobs)(delayed(_core_batch)(root, items, segs[k], len(k) == 1, cache)
                                for k, items in batches.items())
    res = [r for b in out for r in b]
    if timer is not None:
        timer.add(f'core {df.dataset.iloc[0]}: поиск тканей + маска сегментатора + отказы + признаки',
                  [t for *_, t in res], time.perf_counter() - t0)
    return [{'i': i, 'role': role, **x} for i, role, x, _ in res]


def route_counts(J, n):
    """Сколько снимков (role 'test') анализ фото примет какой моделью и сколько отклонит, по причинам."""
    T = [x for x in J if x['role'] == 'test']
    c = {'n_total': n}
    for x in T:
        k = x['key'] or 'отказ · ' + x['why'].split(':')[0]
        c[k] = c.get(k, 0) + 1
    return c


def _frame(J, role, key=None, s=False):
    sel = [x for x in J if x['role'] == role and x['key'] is not None and (key is None or x['key'] == key)]
    return pd.DataFrame([r for x in sel for r in (x['srows'] if s else x['rows'])])


def run_core(df, J, key):
    """OOF ядра модели key на снимках, которые анализ фото оценит этой моделью; модель фолда f учится на
    принятых снимках остальных фолдов с масками сегментаторов, не видевших фолда f (nested в core_jobs).
    → (метрики на принятых: OOD > OOD_MAX — тоже отказ; OOF-таблица всех оценённых этой моделью снимков)."""
    F = _frame(J, 'test', key)
    if not len(F):
        return None, None                            # анализ фото ни один снимок этой моделью не оценит
    F = CORE.usable(F, key)
    feats = CORE.feature_list(F.columns, key)
    S = _frame(J, 'test', key, s=True)
    S = S[S.group.isin(set(F.group))] if len(S) else None
    folds = sorted(set(df.fold.astype(int)))
    train = {f: CORE.usable(_frame(J, f), key) for f in folds}
    O, SP = CORE.cross_validate(F, feats, key, S, train_by_fold=train)
    acc = O.ood.values <= CORE.OOD_MAX
    r = CORE.core_metrics(O[acc], None if SP is None else SP[SP.group.isin(set(O.group[acc]))])
    r.update(target_sens=CORE.SENS_TARGET, n_routed=int(len(O)), n_ood=int((~acc).sum()), n_accepted=int(acc.sum()),
             n_total=len(df))
    return r, O.assign(accepted=acc)


# ------------------------------------------------------------------ ядро: проверка между странами (Eyes-defy)
COUNTRY_DIRECTIONS = (('India', 'Italy'), ('Italy', 'India'))


def country_of(df):
    """Страна пациента Eyes-defy из patient_id (ED_<страна>_<номер>); у других датасетов страны нет."""
    assert (df.dataset == 'eyes_defy').all(), 'страна есть только у Eyes-defy'
    c = df.patient_id.str.split('_').str[1]
    assert c.isin([x for d in COUNTRY_DIRECTIONS for x in d]).all(), 'неизвестная страна в patient_id'
    return c


def country_split(df, train_c, test_c):
    """→ (A — снимки страны обучения, B — снимки страны теста, исключённые группы). Группа манифеста (пациент +
    подтверждённые копии снимка), попавшая в обе страны, — это копия одного снимка: она убирается из обеих частей
    и печатается, иначе тест видел бы обучающий снимок."""
    country = country_of(df)
    both = set(df.group[(country == train_c).values]) & set(df.group[(country == test_c).values])
    keep = ~df.group.isin(both).values
    A = df[(country == train_c).values & keep].reset_index(drop=True)
    B = df[(country == test_c).values & keep].reset_index(drop=True)
    assert len(A) and len(B), f'нет снимков {train_c} или {test_c}'
    assert not set(A.group) & set(B.group)
    return A, B, sorted(both)


def eval_external(T, F, S, key):
    """Модель key обучается только на T (страна обучения; пороги и решение о Hb — вложенной CV внутри T, OOD — по
    диапазону T) и применяется к F (страна теста) без какой-либо настройки на F.
    → (метрики на принятых снимках или None, OOF-таблица F, строка базовой модели «самый частый класс T»)."""
    if not len(T) or not len(F):                       # анализ фото не оценит этой моделью ни одного снимка
        return None, None, None
    T, F = CORE.usable(T, key), CORE.usable(F, key)
    if not len(T) or not len(F):
        return None, None, None
    T0 = T[T.aug == 0]
    n_cls = T0.drop_duplicates('group').anemic.value_counts()
    if len(n_cls) < 2 or n_cls.min() < 2 * CORE.INNER_FOLDS:   # мало пациентов класса: вложенная CV порогов не соберётся
        return None, None, None
    assert not set(T.group) & set(F.group), 'пациент теста в обучении'
    feats = CORE.feature_list(T.columns, key)
    B = CORE.fit_core(T, feats, key)
    O = F[F.aug == 0].reset_index(drop=True)
    p, lv, hbp, ood = CORE.apply_core(B, O[feats].values)
    O = O.assign(p=p, level=lv, hb_pred=hbp, ood=ood, hb_ok=B['hb_ok'], thr_sens=B['thr_sens'],
                 hb_hw=B['hb_halfwidth'])
    acc = O.ood.values <= CORE.OOD_MAX
    if not acc.any():
        return None, O.assign(accepted=acc), None
    SP = None
    if S is not None and len(S):
        S = S[S.group.isin(set(O.group[acc]))]
        if len(S):
            ps, ls, _, _ = CORE.apply_core(B, S[feats].values)
            SP = S[['group', 'cond', 'anemic']].assign(p=ps, level=ls)
    t0 = T[T.aug == 0]
    r = CORE.core_metrics(O[acc], SP)
    r.update(target_sens=CORE.SENS_TARGET, n_train=int(t0.group.nunique()), n_routed=int(len(O)),
             n_ood=int((~acc).sum()), n_accepted=int(acc.sum()), train_prev=float(t0.anemic.mean()),
             test_prev=float(O.anemic.values[acc].mean()))
    const = bool(t0.anemic.mean() >= 0.5)              # простая базовая модель: без фото, только частота в T
    y = O.anemic.values[acc].astype(bool)
    base = {'n': int(acc.sum()), 'const': const, **M.binary_report(np.full(len(y), const), y, O.group.values[acc])}
    base['sens'], base['sens_lo'], base['sens_hi'] = base['rec'], base['rec_lo'], base['rec_hi']
    return r, O.assign(accepted=acc), base


def run_country(root, df, jobs, cache=NO_CACHE, timer=None, external=True):
    """Внешняя проверка внутри Eyes-defy: все пациенты одной страны → обучение, все пациенты другой → тест.
    Всё обучаемое видит только страну обучения: сегментатор конъюнктивы по маскам врачей, модель риска, её пороги,
    диапазон OOD, решение «показывать Hb». Маски обучающих снимков — сегментатором без их фолда (OOF внутри страны
    обучения, как в train_core), тестовых — сегментатором на всей стране обучения (как predict на новом фото).
    → [(направление, key, метрики или None, OOF или None, строка базовой модели или None)]."""
    out = []
    for train_c, test_c in COUNTRY_DIRECTIONS:
        A, B, dropped = country_split(df, train_c, test_c)
        if dropped:
            print(f'{train_c}→{test_c}: копии снимков в обеих странах исключены из обеих частей: {len(dropped)} групп '
                  f'({", ".join(dropped[:5])})')
        segs = conj_segmenters(root, A, jobs, cache, timer, nested=False, full=True, external=external)
        JA = core_jobs(root, A, jobs, segs, cache, timer)
        JB = core_jobs(root, B, jobs, {(int(g),): segs[()] for g in set(B.fold)}, cache, timer)
        for key in CORE.specs_for('eyes_defy'):
            S = _frame(JB, 'test', key, s=True)
            r, O, base = eval_external(_frame(JA, 'test', key), _frame(JB, 'test', key), S if len(S) else None, key)
            out.append((f'{train_c}→{test_c}', key, r, O, base))
    return out


def print_binary_table(rows):
    """Бинарный ответ «анемия» при пороге thr_sens (подобран на обучающих пациентах): матрица ошибок и метрики."""
    print(f'\nПоложительный класс: {M.POSITIVE_CLASS}. Ответ «анемия»: порог P(анемия), подобранный только на '
          'обучающих пациентах (у строк v12 «argmax» — класс с наибольшей вероятностью). Accuracy при дисбалансе '
          'классов обманчива — смотреть вместе со сбаланс. точностью и F1. ДИ долей — Уилсон по снимкам (у Eyes-defy '
          '218 снимков на 217 пациентов), F1 и сбаланс. точности — бутстрэп по пациентам.')
    head = ['Задача', 'Датасет', 'Метод', 'n', 'TP / FP / FN / TN', 'Accuracy, %', 'Precision (PPV), %',
            'Recall (чувств.), %', 'Спец., %', 'NPV, %', 'F1', 'Сбаланс. точн. (2 класса), %', 'AUC']
    print('| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for r in rows:
        cells = [r['task'], r['dataset'], r['method'], r['n'], f"{r['tp']} / {r['fp']} / {r['fn']} / {r['tn']}",
                 fmt(r, 'acc', 0, True), fmt(r, 'prec', 0, True), fmt(r, 'rec', 0, True), fmt(r, 'spec', 0, True),
                 fmt(r, 'npv', 0, True), fmt(r, 'f1', 2), fmt(r, 'bacc2', 0, True), fmt(r, 'auc', 3)]
        print('| ' + ' | '.join(str(c) for c in cells) + ' |')


def print_core_table(rows):
    head = ['Датасет', 'n', 'AUC', 'Низкий / средний / высокий, %', 'Верно среди уверенных, %',
            'Больных среди «высокий», %', 'Здоровых среди «низкий», %', 'Больных в «низкий»',
            'Чувств. / спец. @0.80, %', 'MAE Hb, г/л', 'Hb показан, %', 'Hb в диапазоне, %', 'OOD, %', 'Свет: уровень меняется, %']
    print('\n| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for r in rows:
        zones = ' / '.join(f"{r['share_' + k] * 100:.0f}" for k in ('low', 'mid', 'high'))
        cells = [r['dataset'], r['n'], fmt(r, 'auc', 3), zones, fmt(r, 'acc_sure', 0, True),
                 fmt(r, 'high_ppv', 0, True), fmt(r, 'low_npv', 0, True), r['missed_in_low'],
                 f"{fmt(r, 'sens', 0, True)} / {fmt(r, 'spec', 0, True)}", f"{r['mae']:.1f}",
                 fmt(r, 'hb_shown', 0, True), fmt(r, 'hb_range_cover', 0, True), fmt(r, 'ood_share', 0, True), fmt(r, 'light_level_change', 0, True)]
        print('| ' + ' | '.join(str(c) for c in cells) + ' |')


# ------------------------------------------------------------------ вывод
def fmt(r, k, nd=1, pct=False):
    v = r.get(k)
    if v is None or v != v:
        return '—'
    v = v * 100 if pct else v
    lo, hi = r.get(k + '_lo'), r.get(k + '_hi')
    s = f'{v:.{nd}f}'
    if lo is not None and lo == lo:
        lo, hi = (lo * 100, hi * 100) if pct else (lo, hi)
        s += f' [{lo:.{nd}f}–{hi:.{nd}f}]'
    return s


def print_table(rows):
    head = ['Датасет', 'Метод', 'n', 'MAE, г/л', 'Bland–Altman: смещение [LoA]', 'AUC', 'Чувств., %', 'Спец., %',
            'Чувств. Hb<100, %', 'Свет: ±г/л медиана (p90)', 'IoU', 'Dice', 'Найдено, %']
    print('\n| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for r in rows:
        ba = f"{r['bias']:+.1f} [{r['loa_lo']:.0f}; {r['loa_hi']:.0f}]" if 'bias' in r else '—'
        light = f"±{r['light_med']:.1f} ({r['light_p90']:.1f})" if 'light_med' in r else '—'
        hb100 = f"{r['sens_hb100'] * 100:.0f} (n={r['n_hb100']})" if r.get('n_hb100') else '—'
        iou = f"{r['iou_mean']:.3f} [{r['iou_lo']:.3f}–{r['iou_hi']:.3f}]" if 'iou_mean' in r else '—'
        cells = [r['dataset'], r['method'], r['n'], fmt(r, 'mae'), ba, fmt(r, 'auc', 3), fmt(r, 'sens', 0, True),
                 fmt(r, 'spec', 0, True), hb100, light, iou, fmt(r, 'dice_mean', 3), fmt(r, 'found', 0, True)]
        print('| ' + ' | '.join(str(c) for c in cells) + ' |')


def print_v12_table(rows):
    head = ['Датасет', 'Метод', 'n', 'Верный класс (сбаланс.), %', 'AUC', 'Чувств., %', 'Спец., %',
            'AUC при другом свете', 'Свет: ответ меняется, %']
    print('\n| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for r in rows:
        cells = [r['dataset'], r['method'], r['n'], fmt(r, 'bal_acc', 0, True), fmt(r, 'auc', 3),
                 fmt(r, 'sens', 0, True), fmt(r, 'spec', 0, True), fmt(r, 'auc_light', 3), fmt(r, 'light_flip', 0, True)]
        print('| ' + ' | '.join(str(c) for c in cells) + ' |')


def append_history(rows, tag, commit, path=HISTORY, run_id=None, backup_dir=None):
    """Строки результата → CSV истории. Колонки файла не совпали — сначала резервная копия (anemia/runs.py)."""
    cols = HIST_COLS if run_id is None else ['run_id'] + HIST_COLS
    fixed = {'timestamp': dt.datetime.now().isoformat(timespec='seconds'), 'commit': commit, 'tag': tag,
             'run_id': run_id}
    RUNS.append_history(path, rows, cols, fixed, backup_dir or os.path.join(os.path.dirname(path), 'backups'))


def save_oof(O, cols, name, run_dir, oof_dir, backup_dir):
    """OOF: неизменяемая копия в папке запуска + «последний прогон» <data>/results/oof_<tag>_*.csv
    (его читает /error-analysis).
    Последний перезаписывается, но если его копии нет ни в одной папке запуска, он сначала уходит в backups/."""
    O[cols].to_csv(os.path.join(run_dir, name), index=False)
    latest = os.path.join(oof_dir, name)
    RUNS.preserve_before_overwrite(latest, os.path.dirname(run_dir), backup_dir)
    O[cols].to_csv(latest, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('ANEMIA_DATA', 'data'))
    ap.add_argument('--task', choices=['all', 'seg', 'hb', 'v12', 'core', 'country', 'seg_ext'], default='all')
    ap.add_argument('--dataset', choices=['all', 'eyes_defy', 'phone'], default='all')
    ap.add_argument('--model', choices=['hgb', 'ridge', 'both'], default='both')
    ap.add_argument('--target-sens', type=float, default=0.80)
    ap.add_argument('--tag', default='baseline_v11')
    ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--no-history', action='store_true', help='не писать в metrics_history.csv (смоук-тесты)')
    ap.add_argument('--history', default=HISTORY,
                    help='история в репозитории (по умолчанию results/metrics_history.csv)')
    ap.add_argument('--cache-dir', help='кэш признаков (по умолчанию <data>/cache/features)')
    ap.add_argument('--no-cache', action='store_true', help='не читать и не писать кэш признаков')
    ap.add_argument('--no-external-seg', action='store_true',
                    help='сегментатор только по маскам Eyes-defy (без телефонных фото external/, как до 2026-10-20)')
    ap.add_argument('--refresh-cache', action='store_true', help='пересчитать признаки и перезаписать кэш')
    a = ap.parse_args()
    if a.no_cache and a.refresh_cache:
        ap.error('--no-cache и --refresh-cache вместе не имеют смысла')

    t0, started = time.time(), dt.datetime.now().isoformat(timespec='seconds')
    res_dir = os.path.join(a.data, 'results')
    backup_dir = os.path.join(res_dir, 'backups')
    run_id, run_dir = RUNS.create_run_dir(os.path.join(res_dir, 'runs'), a.tag)
    os.makedirs(os.path.join(ROOT, 'results', 'logs'), exist_ok=True)
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    sys.stdout = Tee(os.path.join(ROOT, 'results', 'logs', f'evaluate_{a.tag}_{stamp}.log'),
                     os.path.join(run_dir, 'log.txt'))
    man_path = os.path.join(a.data, 'manifest.csv')
    man = pd.read_csv(man_path)
    git = git_state()
    commit = git['commit']
    script = source_version(os.path.abspath(__file__))
    cache_mode = 'off' if a.no_cache else 'refresh' if a.refresh_cache else 'on'
    cache = FeatureCache(a.cache_dir or os.path.join(a.data, 'cache', 'features'), cache_mode,
                         salt={'evaluate_py': script})
    timer = StageTimer()
    meta = {'run_id': run_id, 'tag': a.tag, 'status': 'running', 'started': started, 'commit': commit, 'git': git,
            'evaluate_py_sha1': script, 'args': vars(a), 'python': sys.version.split()[0],
            'manifest': {'path': man_path, 'sha1': file_sha1(man_path), 'rows': len(man)},
            'cache': {'mode': cache_mode, 'dir': cache.root},
            'params': {'work_size': WORK_SIZE, 'n_relight': N_RELIGHT, 'stress': STRESS_CONDITIONS,
                       'v12': {'detect_side': V12.DETECT_SIDE, 'crop_side': V12.CROP_SIDE, 'n_relight': V12.N_RELIGHT},
                       'seg': {'seg_size': SB.SEG_SIZE, 'pix_per_image': SB.PIX_PER_IMAGE},
                       'target_sens': a.target_sens}}
    RUNS.write_json(os.path.join(run_dir, 'run.json'), meta)
    print(f'evaluate.py · commit {commit} · tag {a.tag} · {stamp}')
    print(f'Запуск {run_id} → {run_dir} · кэш признаков: {cache_mode} ({cache.root})')
    models = ['hgb', 'ridge'] if a.model == 'both' else [a.model]
    os.makedirs(res_dir, exist_ok=True)
    rows = []
    n_ext = 0 if a.no_external_seg or a.task not in ('core', 'country') else len(EXT.seg_index(a.data))
    seg_note = f' · сегментатор +{n_ext} тел. фото' if n_ext else ''   # строки с/без телефонных фото различимы
    for ds in (['eyes_defy', 'phone'] if a.dataset == 'all' else [a.dataset]):
        df = man[man.dataset == ds].reset_index(drop=True)
        if not len(df):
            continue
        if a.task == 'country':
            if ds != 'eyes_defy':
                print(f'{ds}: проверка между странами только для Eyes-defy — пропуск')
                continue
            print(f'[{time.time() - t0:5.0f} с] {ds}: проверка между странами (Индия ↔ Италия)...')
            for direction, key, r, O, base in run_country(a.data, df, a.jobs, cache, timer,
                                                                   external=not a.no_external_seg):
                name = f'{ds} {direction}'
                if r is None:
                    print(f'{name} · {key}: в одной из стран нет снимков для этой модели или для обучения мало пациентов / один класс — пропуск')
                    continue
                rows.append({'task': 'country', 'dataset': name, 'method': f"{key} · core · обучение "
                             f"{r['n_train']} пац., тест принято {r['n_accepted']}/{len(O)} (OOD {r['n_ood']}){seg_note}", **r})
                rows.append({'task': 'country', 'dataset': name, 'method': f"базовая: всегда «"
                             f"{'анемия' if base['const'] else 'норма'}» (частый класс страны обучения)", **base})
                save_oof(O, ['group', 'hb_g_l', 'anemic', 'p', 'level', 'hb_pred', 'ood', 'thr_sens', 'accepted'],
                         f"oof_{a.tag}_{ds}_{direction.replace('→', '_to_')}_{key}_country.csv", run_dir, res_dir,
                         backup_dir)
            continue
        if a.task == 'seg_ext':
            if ds != 'eyes_defy':   # телефонные фото с масками проверяются один раз, рядом с Eyes-defy
                continue
            print(f'[{time.time() - t0:5.0f} с] {ds}: сегментатор ядра без/с телефонными фото, IoU вне обучения...')
            rows += run_seg_external(a.data, df, a.jobs, cache, timer)
            continue
        if a.task == 'core':
            if not df.mask_conj.notna().all():
                sys.exit(f'{ds}: у части снимков нет масок врачей — сегментатор конъюнктивы ядра обучить не на чем')
            print(f'[{time.time() - t0:5.0f} с] {ds}: сегментаторы конъюнктивы по маскам врачей (без тестового фолда)...')
            segs = conj_segmenters(a.data, df, a.jobs, cache, timer, nested=True, full=False,
                                   external=not a.no_external_seg)
            print(f'[{time.time() - t0:5.0f} с] {ds}: маршрут анализа фото, отказы и признаки...')
            J = core_jobs(a.data, df, a.jobs, segs, cache, timer, nested=True)
            rc = route_counts(J, len(df))
            print(f'{ds}: маршрут анализа фото (тестовые маски): ' + ', '.join(f'{k} {v}' for k, v in rc.items()))
            for key in CORE.specs_for(ds):   # у взрослых — модель на каждый набор тканей, одни и те же фолды
                print(f'[{time.time() - t0:5.0f} с] {key}: ядро HemaVision...')
                r, O = run_core(df, J, key)
                if r is None:
                    print(f'{key}: анализ фото не оценит этой моделью ни одного снимка — пропуск')
                    continue
                rows.append({'task': 'core', 'dataset': key, 'method': f"core · принято {r['n_accepted']}/{len(df)} "
                             f"(этой моделью {r['n_routed']}, OOD {r['n_ood']}){seg_note}", **r})
                save_oof(O, ['group', 'fold', 'hb_g_l', 'anemic', 'p', 'level', 'hb_pred', 'ood', 'thr_sens',
                             'accepted'], f'oof_{a.tag}_{key}_core.csv', run_dir, res_dir, backup_dir)
            RUNS.write_json(os.path.join(run_dir, f'routes_{ds}.json'), rc)
            continue
        if a.task in ('all', 'v12'):
            print(f'[{time.time() - t0:5.0f} с] {ds}: модель v12, {len(df)} снимков...')
            for mode, rule, n_skip, r, O in run_v12(a.data, df, a.target_sens, a.jobs, cache, timer):
                method = f'v12 · {mode} · {rule}' + (f' (−{n_skip})' if n_skip else '')
                rows.append({'task': 'v12', 'dataset': ds, 'method': method, **r})
                if rule == 'argmax':
                    P = np.stack(O.P.values)
                    save_oof(O.assign(p_norm=P[:, 0], p_mild=P[:, 1], p_mod=P[:, 2]),
                             ['group', 'fold', 'y', 'anemic', 'p_norm', 'p_mild', 'p_mod', 'thr'],
                             f'oof_{a.tag}_{ds}_v12_{mode}.csv', run_dir, res_dir, backup_dir)
            if a.task == 'v12':
                continue
        has_masks = df.mask_conj.notna().all()
        pred_masks = None
        if has_masks and a.task in ('all', 'seg'):
            print(f'[{time.time() - t0:5.0f} с] {ds}: baseline-сегментация v11, {len(df)} снимков, '
                  f'{df.fold.nunique()} фолдов...')
            res, pred_masks, _ = run_seg(a.data, df, a.jobs, timer)
            rows.append({'task': 'seg', 'dataset': ds, 'method': 'v11 pixel-HGB', **res})
        if a.task not in ('all', 'hb'):
            continue
        sources = [('doctor', None), ('pred', pred_masks)] if has_masks else [('full', None)]
        for src, masks in sources:
            if src == 'pred' and masks is None:
                continue
            for mk in models:
                print(f'[{time.time() - t0:5.0f} с] {ds}: Hb, маска={src}, модель={mk}...')
                r, O, n_skip = run_hb(a.data, df, 'doctor' if src == 'doctor' else src, masks, mk, a.target_sens,
                                      a.jobs, cache, timer, f'Hb {ds}, маска {src}, {mk}: чтение + признаки')
                method = f'{mk} · маска {src}'
                if n_skip:
                    print(f'   ⚠️ без ткани на маске: {n_skip} снимков — исключены из метрик Hb')
                    method += f' (−{n_skip})'
                rows.append({'task': 'hb', 'dataset': ds, 'method': method, **r})
                save_oof(O, ['group', 'fold', 'hb_g_l', 'pred', 'score', 'thr', 'flag', 'anemic'],
                         f'oof_{a.tag}_{ds}_{src}_{mk}.csv', run_dir, res_dir, backup_dir)
    if any(r['task'] in ('seg', 'hb', 'seg_ext') for r in rows):
        print_table([r for r in rows if r['task'] in ('seg', 'hb', 'seg_ext')])
    if any(r['task'] == 'core' for r in rows):
        print_core_table([r for r in rows if r['task'] == 'core'])
    if any(r['task'] == 'country' and 'share_low' in r for r in rows):
        print_core_table([r for r in rows if r['task'] == 'country' and 'share_low' in r])
    if any('tp' in r for r in rows):
        print_binary_table([r for r in rows if 'tp' in r])
    if any(r['task'] == 'v12' for r in rows):
        print_v12_table([r for r in rows if r['task'] == 'v12'])
    total = time.time() - t0
    feat_wall = sum(r['wall'] for r in timer.rows)
    print('\nВремя (чтение / признаки / кэш — суммы по рабочим процессам; «стена» — по часам):')
    print(timer.table())
    print(f'Всего {total:.1f} с, из них этапы чтения и признаков {feat_wall:.1f} с, '
          f'обучение моделей, CV и метрики {total - feat_wall:.1f} с.')
    pd.DataFrame([{'run_id': run_id, **r} for r in rows]).to_csv(os.path.join(run_dir, 'metrics.csv'), index=False)
    if not a.no_history:
        append_history(rows, a.tag, commit, os.path.join(res_dir, 'metrics_history.csv'), run_id, backup_dir)
        append_history(rows, a.tag, commit, a.history, backup_dir=backup_dir)
        print(f'\nСтрок дописано: {len(rows)} → {a.history} и {os.path.join(res_dir, "metrics_history.csv")}')
    meta.update(status='done', finished=dt.datetime.now().isoformat(timespec='seconds'), total_s=round(total, 2),
                timings=timer.rows, n_rows=len(rows),
                history=None if a.no_history else [a.history, os.path.join(res_dir, 'metrics_history.csv')])
    RUNS.write_json(os.path.join(run_dir, 'run.json'), meta)
    print(f'Результаты запуска: {run_dir}')


if __name__ == '__main__':
    main()
