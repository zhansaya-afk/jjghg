"""Метрики масок конъюнктивы по снимкам для отчёта evaluate.py (seg_per_image.csv, seg_summary.csv).

Только отчёт: обучение, признаки и анализ фото этот модуль не вызывают, поэтому его нет в
model_store.FEATURE_MODULES (метка модели от него не зависит).

Статус снимка:
    no_gt   — эталонной маски нет или она некорректна (пустая, в списке испорченных) — метрики не считаются;
    refused — эталон корректен, но система отклонила сегментацию (анатомия, согласие тканей, качество, отказ
              маршрута до сегментатора, модель без конъюнктивы) — в средние метрики не входит;
    ok      — эталон корректен и есть результат сегментации, в том числе пустой: пустая маска при непустом
              эталоне — ошибка сегментации (recall, IoU, Dice = 0; precision не определена — NaN).
Метрика с нулевым знаменателем — NaN, а не 0 и не 1.

Этап (stage) — где решилась судьба маски (колонка seg_per_image.csv и счётчики n_<stage> в seg_summary.csv):
    ok:      accepted — маска сегментатора принята; empty_mask — сегментатор не нашёл веко;
             accepted_no_segmenter — принята маска, которую дал не сегментатор (вырезанная ткань: сегментатор
             не запускается, маска — пороговая regions; так было и в _crop_iou);
    refused: before_seg_inconsistent — ткани шага 1 не складываются в глаз, сегментатор не запускался;
             not_segmented — тканей шага 1 нет, сегментатор не запускался;
             anatomy — маска сегментатора не прошла анатомическую проверку века (AN.check);
             after_seg_inconsistent — маска прошла анатомию, но не согласуется с радужкой (_inconsistent);
             quality — веко принято, снимок отклонён проверкой качества тканей/кадра (tissue_route);
             no_conj_model — снимок оценён моделью без конъюнктивы (склера);
    no_gt:   no_gt.
"""
import numpy as np
import pandas as pd

from . import metrics as M

STATUSES = ('ok', 'refused', 'no_gt')
COUNTRY_PREFIXES = {'ED_India_': 'India', 'ED_Italy_': 'Italy'}
UNKNOWN = 'unknown'
OK_STAGES = ('accepted', 'empty_mask', 'accepted_no_segmenter')
REFUSED_STAGES = ('before_seg_inconsistent', 'not_segmented', 'anatomy', 'after_seg_inconsistent', 'quality',
                  'no_conj_model')
STAGES = {'ok': OK_STAGES, 'refused': REFUSED_STAGES, 'no_gt': ('no_gt',)}
CSV_COLUMNS = ('task', 'arm', 'dataset', 'id', 'group', 'fold', 'country', 'status', 'stage', 'why', 'pred_empty',
               'accepted',
               'gt_px', 'pred_px', 'tp', 'fp', 'fn', 'iou', 'dice', 'precision', 'recall')
METRICS = ('iou', 'dice', 'precision', 'recall')
NAN = float('nan')


def _div(a, b):
    """a / b или NaN при нулевом знаменателе (без предупреждений numpy)."""
    return float(a) / float(b) if b else NAN


def _as_mask(m):
    m = np.asarray(m)
    return m if m.dtype == bool else m > 0


def pixel_scores(pred, gt):
    """Маски одного снимка (bool или 0/255) → пиксельные счётчики и метрики. Пустой эталон → все метрики NaN."""
    pred, gt = _as_mask(pred), _as_mask(gt)
    if pred.shape != gt.shape:
        raise ValueError(f'размеры масок не совпадают: {pred.shape} и {gt.shape}')
    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())
    out = {'gt_px': tp + fn, 'pred_px': tp + fp, 'tp': tp, 'fp': fp, 'fn': fn}
    if tp + fn == 0:                                    # эталона нет — ни одна метрика не определена
        return {**out, **{k: NAN for k in METRICS}}
    return {**out, 'iou': _div(tp, tp + fp + fn), 'dice': _div(2 * tp, 2 * tp + fp + fn),
            'precision': _div(tp, tp + fp), 'recall': _div(tp, tp + fn)}


def country_of_id(image_id):
    """Страна только по известным префиксам patient_id Eyes-defy; всё остальное — 'unknown'."""
    if isinstance(image_id, str):
        for prefix, country in COUNTRY_PREFIXES.items():
            if image_id.startswith(prefix) and len(image_id) > len(prefix):
                return country
    return UNKNOWN


def image_record(task, arm, dataset, image_id, group, fold, status, pred=None, gt=None, accepted=False, why='',
                 stage=None):
    """Строка seg_per_image.csv. pred=None — сегментатор не запускался (только у refused и no_gt).
    У refused метрики пишутся по отклонённой маске (диагностика), в средние они не входят.
    stage — из STAGES[status]; у refused обязателен, у ok по умолчанию accepted / empty_mask, у no_gt — no_gt."""
    if status not in STATUSES:
        raise ValueError(f'неизвестный статус {status!r}')
    if stage is None:
        if status == 'refused':
            raise ValueError('refused: нужен этап отказа (stage)')
        stage = 'no_gt' if status == 'no_gt' else ('accepted' if accepted else 'empty_mask')
    if stage not in STAGES[status]:
        raise ValueError(f'{status}: неизвестный этап {stage!r} (допустимы {STAGES[status]})')
    if status == 'ok' and (stage == 'empty_mask') == bool(accepted):
        raise ValueError(f'ok: этап {stage!r} не согласуется с accepted={bool(accepted)}')
    rec = {'task': task, 'arm': arm, 'dataset': dataset, 'id': str(image_id), 'group': str(group),
           'fold': -1 if fold is None or pd.isna(fold) else int(fold), 'country': country_of_id(image_id),
           'status': status, 'stage': stage, 'why': why or '', 'accepted': bool(accepted),
           'pred_empty': None if pred is None else not _as_mask(pred).any()}
    blank = {'gt_px': NAN, 'pred_px': NAN, 'tp': NAN, 'fp': NAN, 'fn': NAN, **{k: NAN for k in METRICS}}
    if status == 'no_gt':
        if accepted:
            raise ValueError('no_gt: принятой маски без эталона в отчёте не бывает')
        gt_px = int(_as_mask(gt).sum()) if gt is not None else NAN
        return {**rec, **blank, 'gt_px': gt_px}
    if gt is None or not _as_mask(gt).any():
        raise ValueError(f'{status}: нужен непустой эталон (иначе статус no_gt)')
    if status == 'ok':
        if pred is None:
            raise ValueError('ok: нужен результат сегментации (пустая маска — массив без пикселей)')
        if accepted and rec['pred_empty']:
            raise ValueError('ok: пустая маска не может быть принята в анализ')
    elif accepted:
        raise ValueError('refused: отклонённая маска не может быть принята')
    if pred is None:
        return {**rec, **blank, 'gt_px': int(_as_mask(gt).sum())}
    return {**rec, **pixel_scores(pred, gt)}


def records_frame(records):
    """Строки image_record → DataFrame со столбцами ровно CSV_COLUMNS."""
    return pd.DataFrame(list(records), columns=list(CSV_COLUMNS))


def _ci(values, groups):
    if len(values) == 0 or len(set(groups)) < 2:
        return NAN, NAN
    return M.bootstrap_ci(np.mean, groups, 1000, 0, values)


def _reasons(why, k=3):
    heads = [w.split(':')[0].strip() for w in why if isinstance(w, str) and w]
    s = pd.Series(heads).value_counts().head(k) if heads else pd.Series(dtype=int)
    return '; '.join(f'{r} ({n})' for r, n in s.items())


def _summary_row(d):
    ok = d[d.status == 'ok']
    n_ok, n_ref = len(ok), int((d.status == 'refused').sum())
    empty = ok.pred_empty.astype(bool)
    r = {'n_total': len(d), 'n_no_gt': int((d.status == 'no_gt').sum()), 'n_refused': n_ref, 'n_ok': n_ok,
         'n_pred_empty': int(empty.sum())}
    r['refused_share'], r['refused_lo'], r['refused_hi'] = M.wilson(n_ref, n_ok + n_ref)
    r['empty_share'], r['empty_lo'], r['empty_hi'] = M.wilson(int(empty.sum()), n_ok)
    groups = ok.group.values
    for k in ('iou', 'dice', 'recall'):
        v = ok[k].astype(float).values
        r[f'{k}_mean'] = float(v.mean()) if n_ok else NAN
        r[f'{k}_median'] = float(np.median(v)) if n_ok else NAN
        r[f'{k}_lo'], r[f'{k}_hi'] = _ci(v, groups)
    pos = ok[~empty]                                    # precision определена только при непустой маске
    v = pos.precision.astype(float).values
    r['n_precision'] = len(pos)
    r['precision_mean'] = float(v.mean()) if len(pos) else NAN
    r['precision_median'] = float(np.median(v)) if len(pos) else NAN
    r['precision_lo'], r['precision_hi'] = _ci(v, pos.group.values)
    tp, fp, fn = (float(ok[c].sum()) for c in ('tp', 'fp', 'fn'))
    r['precision_pooled'] = _div(tp, tp + fp)
    r['recall_pooled'] = _div(tp, tp + fn)
    for st in OK_STAGES[2:] + REFUSED_STAGES:            # этапы: где теряются снимки (accepted/empty_mask — выше)
        r[f'n_{st}'] = int((d.stage == st).sum())
    r['refused_why'] = _reasons(d.why[d.status == 'refused'])
    return r


def summarize(df):
    """Сводка по (task, arm, dataset) и стране: 'all' + каждая страна, где есть снимки (+ 'unknown', если среди
    снимков с известной страной есть и снимки без неё).
    Средние IoU, Dice, recall — по всем ok (пустые маски — нулями), precision — по ok с непустой маской;
    refused и no_gt в средние не входят и видны только в счётчиках и долях."""
    out = []
    for (task, arm, ds), d in df.groupby(['task', 'arm', 'dataset'], sort=False):
        present = [c for c in ('India', 'Italy', UNKNOWN) if (d.country == c).any()]
        if present == [UNKNOWN]:                        # страны нет ни у одного снимка (телефонные фото) — только 'all'
            present = []
        parts = [('all', d)] + [(c, d[d.country == c]) for c in present]
        for country, part in parts:
            out.append({'task': task, 'arm': arm, 'dataset': ds, 'country': country, **_summary_row(part)})
    return pd.DataFrame(out)
