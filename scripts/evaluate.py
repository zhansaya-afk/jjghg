"""Единственный источник цифр точности. Каждая строка результата дописывается в results/metrics_history.csv.

    python scripts/evaluate.py --data data --tag baseline_v11            # всё, что есть в манифесте
    python scripts/evaluate.py --task hb --dataset cp_anemic --model ridge

Задачи:
  seg  — baseline-сегментация v11 (Eyes-defy): IoU/Dice с масками врачей на невиданных пациентах.
  hb   — Hb по цвету конъюнктивы: MAE, Bland–Altman, AUC, чувствительность/специфичность при пороге,
         подобранном внутри обучающих фолдов, стресс-тест «другой свет».
         Маска: CP-AnemiC — весь вырез; Eyes-defy — маска врача (потолок) и маска baseline-сегментации.
         MAE сравнивается с «нулевой моделью» (медиана Hb обучающих фолдов): skill = 1 − MAE/MAE₀.
Все метрики — по фиксированным фолдам манифеста (по пациенту). OOF-предсказания → <data>/results/.

Было/стало: --compare TAG сравнивает с OOF прошлого прогона TAG на тех же пациентах (парный бутстрэп
по пациентам) и пишет вердикт «лучше / хуже / в пределах шума» — по нему, а не по разнице точечных оценок.
    python scripts/evaluate.py --task hb --relight 3 --tag baseline_v11 --no-history
    python scripts/evaluate.py --task hb --tag relight12 --compare baseline_v11
"""
import argparse
import datetime as dt
import os
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anemia import metrics as M  # noqa: E402
from anemia import segment_baseline as SB  # noqa: E402
from anemia.features import N_RELIGHT, feature_rows, load_item, row_meta  # noqa: E402
from anemia.hb_models import cross_validate  # noqa: E402

HISTORY = os.path.join(ROOT, 'results', 'metrics_history.csv')
HIST_COLS = ['timestamp', 'commit', 'tag', 'task', 'dataset', 'method', 'n',
             'mae', 'mae_lo', 'mae_hi', 'bias', 'loa_lo', 'loa_hi', 'auc', 'auc_lo', 'auc_hi',
             'target_sens', 'sens', 'sens_lo', 'sens_hi', 'spec', 'spec_lo', 'spec_hi',
             'n_hb100', 'sens_hb100', 'light_med', 'light_p90',
             'iou_mean', 'iou_lo', 'iou_hi', 'iou_median', 'dice_mean', 'found',
             'mae_null', 'skill', 'skill_lo', 'skill_hi', 'relight', 'seed',
             'ref_tag', 'd_mae', 'd_mae_lo', 'd_mae_hi', 'd_auc', 'd_auc_lo', 'd_auc_hi', 'verdict']


class Tee:
    def __init__(self, path):
        self.f, self.out = open(path, 'w'), sys.stdout

    def write(self, s):
        self.out.write(s)
        self.f.write(s)

    def flush(self):
        self.out.flush()
        self.f.flush()


def git_commit():
    try:
        h = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True).stdout.strip()
        return h + ('-dirty' if dirty else '')
    except Exception:
        return 'unknown'


# ------------------------------------------------------------------ сегментация
def _seg_prep(root, row):
    rgb, doc = load_item(root, row, 'doctor')
    return rgb, doc, SB.pixel_features(rgb)


def run_seg(root, df, jobs):
    """OOF-маски baseline-сегментации по фолдам манифеста + метрики."""
    items = Parallel(n_jobs=jobs)(delayed(_seg_prep)(root, r) for r in df.to_dict('records'))
    pred = [None] * len(df)
    folds = df.fold.values
    for f in sorted(set(folds)):
        tr = [i for i in range(len(df)) if folds[i] != f]
        model = SB.train([items[i][2] for i in tr], [items[i][1] for i in tr])
        for i in np.flatnonzero(folds == f):
            pred[i] = SB.predict(model, items[i][2])
    ious = np.array([M.iou(p, it[1]) for p, it in zip(pred, items)])
    dices = np.array([M.dice(p, it[1]) for p, it in zip(pred, items)])
    lo, hi = M.bootstrap_ci(np.mean, df.group.values, 1000, 0, ious)
    res = {'n': len(df), 'iou_mean': ious.mean(), 'iou_lo': lo, 'iou_hi': hi, 'iou_median': float(np.median(ious)),
           'dice_mean': dices.mean(), 'found': float(np.mean([p.any() for p in pred]))}
    return res, pred, items


# ------------------------------------------------------------------ Hb
def _feat_job(root, row, mask_source, mask, seed, n_relight):
    if mask is None:
        rgb, mask = load_item(root, row, mask_source)
    else:
        rgb, _ = load_item(root, row, 'full')
    return feature_rows(rgb, mask, row_meta(row), seed, n_relight)


def hb_metrics(O, S, target):
    y, pred, true = O.anemic.values.astype(bool), O.pred.values, O.hb_g_l.values
    g = O.group.values
    r = {'n': len(O), 'mae': M.mae(pred, true), 'target_sens': target, **M.bland_altman(pred, true)}
    r['mae_lo'], r['mae_hi'] = M.bootstrap_ci(M.mae, g, 1000, 0, pred, true)
    null = O.null.values
    r['mae_null'] = M.mae(null, true)
    r['skill'] = M.skill(pred, null, true)
    r['skill_lo'], r['skill_hi'] = M.bootstrap_ci(M.skill, g, 1000, 0, pred, null, true)
    if 0 < y.sum() < len(y):
        r['auc'] = M.auc(O.score.values, y)
        r['auc_lo'], r['auc_hi'] = M.bootstrap_ci(lambda s, t: M.auc(s, t), g, 1000, 0, O.score.values, y)
    flag = O.flag.values
    r['sens'], r['sens_lo'], r['sens_hi'] = M.wilson(int((flag & y).sum()), int(y.sum()))
    r['spec'], r['spec_lo'], r['spec_hi'] = M.wilson(int((~flag & ~y).sum()), int((~y).sum()))
    low = true < 100
    r['n_hb100'] = int(low.sum())
    r['sens_hb100'] = float(flag[low].mean()) if low.any() else float('nan')
    if S is not None and len(S):
        spread = S.groupby('group').pred.agg(lambda p: (p.max() - p.min()) / 2)
        r['light_med'], r['light_p90'] = float(spread.median()), float(spread.quantile(0.9))
    return r


def run_hb(root, df, mask_source, masks, model_kind, target, jobs, n_relight=N_RELIGHT, seed=0):
    out = Parallel(n_jobs=jobs)(delayed(_feat_job)(root, r, mask_source, None if masks is None else masks[i],
                                                   i + 100_000 * seed, n_relight)
                                for i, r in enumerate(df.to_dict('records')))
    F = pd.DataFrame([x for rows, _ in out for x in rows])
    S = pd.DataFrame([x for _, srows in out for x in srows])
    n_skip = len(df) - F[F.aug == 0].group.nunique() if len(F) else len(df)
    feats = [c for c in F.columns if c.startswith('c_')]
    if df.sex.notna().any():
        feats.append('male')
    O, SP = cross_validate(F, feats, model_kind, target, stress=S if len(S) else None)
    return hb_metrics(O, SP, target), O, n_skip


def compare(O, ref_path):
    """Парное сравнение с OOF прошлого прогона на общих пациентах: Δ = новый − старый, 95% ДИ бутстрэпом
    по пациентам. ΔMAE < 0 и ΔAUC > 0 — лучше. Вердикт по ДИ: ноль внутри → «в пределах шума»."""
    R = pd.read_csv(ref_path)
    J = O[['group', 'hb_g_l', 'anemic', 'pred', 'score']].merge(
        R[['group', 'pred', 'score']], on='group', suffixes=('', '_ref'))
    g, t, y = J.group.values, J.hb_g_l.values, J.anemic.values.astype(bool)
    d_mae = lambda p, q, t: M.mae(p, t) - M.mae(q, t)
    d_auc = lambda s, u, y: M.auc(s, y) - M.auc(u, y)
    r = {'n_cmp': len(J), 'd_mae': d_mae(J.pred.values, J.pred_ref.values, t)}
    r['d_mae_lo'], r['d_mae_hi'] = M.bootstrap_ci(d_mae, g, 1000, 0, J.pred.values, J.pred_ref.values, t)
    if 0 < y.sum() < len(y):
        r['d_auc'] = d_auc(J.score.values, J.score_ref.values, y)
        r['d_auc_lo'], r['d_auc_hi'] = M.bootstrap_ci(d_auc, g, 1000, 0, J.score.values, J.score_ref.values, y)
    better = r['d_mae_hi'] < 0 or r.get('d_auc_lo', -1) > 0
    worse = r['d_mae_lo'] > 0 or r.get('d_auc_hi', 1) < 0
    r['verdict'] = 'смешанно' if better and worse else 'лучше' if better else 'хуже' if worse else 'в пределах шума'
    return r


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
    head = ['Датасет', 'Метод', 'n', 'MAE, г/л', 'MAE константы', 'Skill, %', 'Bland–Altman: смещение [LoA]', 'AUC', 'Чувств., %', 'Спец., %',
            'Чувств. Hb<100, %', 'Свет: ±г/л медиана (p90)', 'IoU', 'Dice', 'Найдено, %']
    print('\n| ' + ' | '.join(head) + ' |')
    print('|' + '---|' * len(head))
    for r in rows:
        ba = f"{r['bias']:+.1f} [{r['loa_lo']:.0f}; {r['loa_hi']:.0f}]" if 'bias' in r else '—'
        light = f"±{r['light_med']:.1f} ({r['light_p90']:.1f})" if 'light_med' in r else '—'
        hb100 = f"{r['sens_hb100'] * 100:.0f} (n={r['n_hb100']})" if r.get('n_hb100') else '—'
        iou = f"{r['iou_mean']:.3f} [{r['iou_lo']:.3f}–{r['iou_hi']:.3f}]" if 'iou_mean' in r else '—'
        cells = [r['dataset'], r['method'], r['n'], fmt(r, 'mae'), fmt(r, 'mae_null'), fmt(r, 'skill', 0, True), ba, fmt(r, 'auc', 3), fmt(r, 'sens', 0, True),
                 fmt(r, 'spec', 0, True), hb100, light, iou, fmt(r, 'dice_mean', 3), fmt(r, 'found', 0, True)]
        print('| ' + ' | '.join(str(c) for c in cells) + ' |')
    cmp = [r for r in rows if r.get('verdict')]
    if cmp:
        print(f"\nБыло/стало относительно «{cmp[0]['ref_tag']}» (те же пациенты; Δ = новый − старый, 95% ДИ):")
        print('| Датасет | Метод | n | ΔMAE, г/л | ΔAUC | Вердикт |\n|---|---|---|---|---|---|')
        for r in cmp:
            print(f"| {r['dataset']} | {r['method']} | {r['n_cmp']} | {fmt(r, 'd_mae', 2)} | {fmt(r, 'd_auc', 3)} | "
                  f"{r['verdict']} |")


def append_history(rows, tag, commit):
    os.makedirs(os.path.dirname(HISTORY), exist_ok=True)
    ts = dt.datetime.now().isoformat(timespec='seconds')
    H = pd.DataFrame([{**{c: None for c in HIST_COLS}, 'timestamp': ts, 'commit': commit, 'tag': tag,
                       **{k: v for k, v in r.items() if k in HIST_COLS}} for r in rows])[HIST_COLS]
    if os.path.exists(HISTORY):
        old = pd.read_csv(HISTORY, dtype=str, keep_default_na=False)
        if list(old.columns) != HIST_COLS:       # добавились колонки: переписать файл с новой шапкой
            old.reindex(columns=HIST_COLS, fill_value='').to_csv(HISTORY, index=False)
    H.to_csv(HISTORY, mode='a', header=not os.path.exists(HISTORY), index=False, float_format='%.4f')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('ANEMIA_DATA', 'data'))
    ap.add_argument('--task', choices=['all', 'seg', 'hb'], default='all')
    ap.add_argument('--dataset', choices=['all', 'eyes_defy', 'cp_anemic', 'phone'], default='all')
    ap.add_argument('--model', choices=['hgb', 'ridge', 'both'], default='both')
    ap.add_argument('--target-sens', type=float, default=0.80)
    ap.add_argument('--tag', default='baseline_v11')
    ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--relight', type=int, default=N_RELIGHT,
                    help=f'обучающих копий при случайном свете на снимок (по умолчанию {N_RELIGHT}; база v11 — 3)')
    ap.add_argument('--seed', type=int, default=0, help='seed случайного света: разброс между seed = шум метода')
    ap.add_argument('--compare', metavar='TAG', help='сравнить с OOF прогона TAG из <data>/results/ (было/стало)')
    ap.add_argument('--no-history', action='store_true', help='не писать в metrics_history.csv (смоук-тесты)')
    a = ap.parse_args()

    os.makedirs(os.path.join(ROOT, 'results', 'logs'), exist_ok=True)
    stamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    sys.stdout = Tee(os.path.join(ROOT, 'results', 'logs', f'evaluate_{a.tag}_{stamp}.log'))
    man = pd.read_csv(os.path.join(a.data, 'manifest.csv'))
    commit = git_commit()
    print(f'evaluate.py · commit {commit} · tag {a.tag} · relight {a.relight} · seed {a.seed} · {stamp}')
    models = ['hgb', 'ridge'] if a.model == 'both' else [a.model]
    oof_dir = os.path.join(a.data, 'results')
    os.makedirs(oof_dir, exist_ok=True)
    rows, t0 = [], time.time()
    for ds in (['eyes_defy', 'cp_anemic', 'phone'] if a.dataset == 'all' else [a.dataset]):
        df = man[man.dataset == ds].reset_index(drop=True)
        if not len(df):
            continue
        has_masks = df.mask_conj.notna().all()
        pred_masks = None
        if has_masks and a.task in ('all', 'seg'):
            print(f'[{time.time() - t0:5.0f} с] {ds}: baseline-сегментация v11, {len(df)} снимков, '
                  f'{df.fold.nunique()} фолдов...')
            res, pred_masks, _ = run_seg(a.data, df, a.jobs)
            rows.append({'task': 'seg', 'dataset': ds, 'method': 'v11 pixel-HGB', **res})
        if a.task not in ('all', 'hb'):
            continue
        sources = [('doctor', None), ('pred', pred_masks)] if has_masks else [('full', None)]
        for src, masks in sources:
            if src == 'pred' and masks is None:
                continue
            for mk in models:
                print(f'[{time.time() - t0:5.0f} с] {ds}: Hb, маска={src}, модель={mk}...')
                r, O, n_skip = run_hb(a.data, df, 'doctor' if src == 'doctor' else src, masks, mk, a.target_sens, a.jobs,
                                      a.relight, a.seed)
                r.update(relight=a.relight, seed=a.seed)
                if a.compare:
                    ref = os.path.join(oof_dir, f'oof_{a.compare}_{ds}_{src}_{mk}.csv')
                    if os.path.exists(ref):
                        r.update(ref_tag=a.compare, **compare(O, ref))
                    else:
                        print(f'   ⚠️ нет {ref}: сравнение пропущено')
                method = f'{mk} · маска {src}'
                if n_skip:
                    print(f'   ⚠️ без ткани на маске: {n_skip} снимков — исключены из метрик Hb')
                    method += f' (−{n_skip})'
                rows.append({'task': 'hb', 'dataset': ds, 'method': method, **r})
                O[['group', 'fold', 'hb_g_l', 'pred', 'null', 'score', 'thr', 'flag', 'anemic']].to_csv(
                    os.path.join(oof_dir, f'oof_{a.tag}_{ds}_{src}_{mk}.csv'), index=False)
    print_table(rows)
    if not a.no_history:
        append_history(rows, a.tag, commit)
        print(f'\nСтрок дописано в results/metrics_history.csv: {len(rows)} · {time.time() - t0:.0f} с')


if __name__ == '__main__':
    main()
