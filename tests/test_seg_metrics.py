"""Метрики масок по снимкам (anemia.seg_metrics): статусы no_gt / refused / ok, нулевые знаменатели, страны.
Синтетика: проверяется устройство кода, а не точность сегментации."""
import re
import warnings

import numpy as np
import pandas as pd
import pytest

from anemia import metrics as M
from anemia import seg_metrics as SM


def box(y0, y1, x0, x1, h=20, w=20):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_pixel_scores_known_counts_match_metrics():
    gt, pred = box(0, 10, 0, 10), box(5, 15, 0, 10)          # пересечение 50, 50 лишних, 50 пропущенных
    s = SM.pixel_scores(pred, gt)
    assert (s['tp'], s['fp'], s['fn'], s['gt_px'], s['pred_px']) == (50, 50, 50, 100, 100)
    assert s['iou'] == pytest.approx(50 / 150) == pytest.approx(M.iou(pred, gt))
    assert s['dice'] == pytest.approx(100 / 200) == pytest.approx(M.dice(pred, gt))
    assert s['precision'] == pytest.approx(0.5) and s['recall'] == pytest.approx(0.5)


def test_empty_prediction_is_segmentation_error():
    s = SM.pixel_scores(np.zeros((20, 20), bool), box(0, 10, 0, 10))
    assert s['iou'] == 0 and s['dice'] == 0 and s['recall'] == 0
    assert np.isnan(s['precision'])                            # 0/0 — не 0 и не 1


def test_empty_reference_gives_nan_without_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for pred in (np.zeros((20, 20), bool), box(0, 5, 0, 5)):
            s = SM.pixel_scores(pred, np.zeros((20, 20), bool))
            assert all(np.isnan(s[k]) for k in SM.METRICS)    # M.iou дал бы 1.0 на пусто/пусто


def test_uint8_masks_and_shape_check():
    gt, pred = box(0, 10, 0, 10), box(5, 15, 0, 10)
    assert SM.pixel_scores(pred.astype(np.uint8) * 255, gt.astype(np.uint8)) == SM.pixel_scores(pred, gt)
    with pytest.raises(ValueError):
        SM.pixel_scores(np.zeros((10, 10), bool), gt)


def test_image_record_statuses():
    gt, pred = box(0, 10, 0, 10), box(0, 10, 0, 5)
    ok = SM.image_record('t', 'a', 'eyes_defy', 'ED_India_1', 'g1', 0, 'ok', pred=pred, gt=gt, accepted=True)
    assert ok['status'] == 'ok' and ok['iou'] == pytest.approx(0.5) and ok['pred_empty'] is False
    empty = SM.image_record('t', 'a', 'eyes_defy', 'ED_India_2', 'g2', 0, 'ok', pred=np.zeros_like(gt), gt=gt)
    assert empty['pred_empty'] is True and empty['iou'] == 0 and np.isnan(empty['precision'])
    assert ok['stage'] == 'accepted' and empty['stage'] == 'empty_mask'
    ref = SM.image_record('t', 'a', 'eyes_defy', 'ED_Italy_3', 'g3', 1, 'refused', gt=gt, why='пятно',
                          stage='not_segmented')
    assert ref['pred_empty'] is None and np.isnan(ref['iou']) and ref['gt_px'] == 100 and ref['stage'] == 'not_segmented'
    no = SM.image_record('t', 'a', 'eyes_defy', 'ED_Italy_4', 'g4', np.nan, 'no_gt')
    assert no['fold'] == -1 and np.isnan(no['iou']) and np.isnan(no['gt_px']) and no['stage'] == 'no_gt'
    with pytest.raises(ValueError):
        SM.image_record('t', 'a', 'd', 'x', 'g', 0, 'found')                         # неизвестный статус
    with pytest.raises(ValueError):
        SM.image_record('t', 'a', 'd', 'x', 'g', 0, 'ok', pred=pred, gt=np.zeros_like(gt))   # ok без эталона
    with pytest.raises(ValueError):
        SM.image_record('t', 'a', 'd', 'x', 'g', 0, 'ok', gt=gt)                     # ok без маски
    with pytest.raises(ValueError):
        SM.image_record('t', 'a', 'd', 'x', 'g', 0, 'ok', pred=np.zeros_like(gt), gt=gt, accepted=True)
    with pytest.raises(ValueError):
        SM.image_record('t', 'a', 'd', 'x', 'g', 0, 'refused', pred=pred, gt=gt, accepted=True, stage='anatomy')
    with pytest.raises(ValueError):
        SM.image_record('t', 'a', 'd', 'x', 'g', 0, 'no_gt', accepted=True)


def test_image_record_stages_validated():
    gt, pred = box(0, 10, 0, 10), box(0, 10, 0, 5)
    with pytest.raises(ValueError, match='этап'):
        SM.image_record('t', 'a', 'd', 'x', 'g', 0, 'refused', pred=pred, gt=gt)              # этап отказа обязателен
    with pytest.raises(ValueError, match='этап'):
        SM.image_record('t', 'a', 'd', 'x', 'g', 0, 'refused', pred=pred, gt=gt, stage='accepted')
    with pytest.raises(ValueError, match='этап'):
        SM.image_record('t', 'a', 'd', 'x', 'g', 0, 'ok', pred=pred, gt=gt, accepted=True, stage='empty_mask')
    with pytest.raises(ValueError, match='этап'):
        SM.image_record('t', 'a', 'd', 'x', 'g', 0, 'ok', pred=np.zeros_like(gt), gt=gt, stage='accepted')
    r = SM.image_record('t', 'a', 'd', 'x', 'g', 0, 'ok', pred=pred, gt=gt, accepted=True,
                        stage='accepted_no_segmenter')
    assert r['status'] == 'ok' and r['stage'] == 'accepted_no_segmenter'
    for st in SM.REFUSED_STAGES:
        assert SM.image_record('t', 'a', 'd', 'x', 'g', 0, 'refused', gt=gt, stage=st)['stage'] == st


def _records():
    gt = box(0, 10, 0, 10)
    rec = lambda i, status, pred=None, **k: SM.image_record(
        'seg_ext', 'arm', 'eyes_defy', f'ED_India_{i}', f'g{i}', 0, status, pred=pred, gt=gt, **k)
    return [rec(1, 'ok', box(0, 6, 0, 10), accepted=True),          # IoU 0,6, precision 1
            rec(2, 'ok', box(0, 8, 0, 10), accepted=True),          # IoU 0,8
            rec(3, 'ok', np.zeros_like(gt)),                        # пустая маска: IoU 0, precision NaN
            rec(4, 'refused', box(10, 20, 0, 10), why='пятно: высота/ширина 2', stage='anatomy'),
            SM.image_record('seg_ext', 'arm', 'eyes_defy', 'ED_India_5', 'g5', 0, 'no_gt', why='нет маски')]


def test_summary_separates_statuses():
    S = SM.summarize(SM.records_frame(_records()))
    r = S[S.country == 'all'].iloc[0]
    assert (r['n_total'], r['n_ok'], r['n_refused'], r['n_no_gt'], r['n_pred_empty']) == (5, 3, 1, 1, 1)
    assert r['iou_mean'] == pytest.approx((0.6 + 0.8 + 0) / 3)       # отказ и снимок без эталона не нулями
    assert r['recall_mean'] == pytest.approx((0.6 + 0.8 + 0) / 3)
    assert r['n_precision'] == 2 and r['precision_mean'] == pytest.approx(1.0)
    assert r['refused_share'] == pytest.approx(1 / 4)                # доля среди снимков с эталоном
    assert r['empty_share'] == pytest.approx(1 / 3)
    assert r['refused_why'] == 'пятно (1)'
    assert r['n_anatomy'] == 1 and r['n_accepted_no_segmenter'] == 0
    assert sum(r[f'n_{st}'] for st in SM.REFUSED_STAGES) == r['n_refused']     # этапы делят все отказы


def test_pooled_and_no_ok_rows():
    S = SM.summarize(SM.records_frame(_records()))
    r = S[S.country == 'all'].iloc[0]
    assert r['precision_pooled'] == pytest.approx(140 / 140)
    assert r['recall_pooled'] == pytest.approx(140 / 300)
    only_bad = [x for x in _records() if x['status'] != 'ok']
    r = SM.summarize(SM.records_frame(only_bad)).iloc[0]
    assert r['n_ok'] == 0 and r['refused_share'] == pytest.approx(1.0)
    for k in ('iou_mean', 'iou_lo', 'precision_mean', 'precision_pooled', 'recall_pooled', 'empty_share'):
        assert np.isnan(r[k])


def test_bootstrap_ci_by_patient():
    recs = _records()
    S = SM.summarize(SM.records_frame(recs))
    r = S[S.country == 'all'].iloc[0]
    assert r['iou_lo'] <= r['iou_mean'] <= r['iou_hi']
    one = SM.summarize(SM.records_frame(recs[:1])).iloc[0]          # один пациент — ДИ не определён
    assert one['iou_mean'] == pytest.approx(0.6) and np.isnan(one['iou_lo']) and np.isnan(one['iou_hi'])


@pytest.mark.parametrize('image_id, country', [
    ('ED_India_12', 'India'), ('ED_Italy_3', 'Italy'), ('ED_india_1', 'unknown'), ('ED_Indiana_1', 'unknown'),
    ('XED_India_1', 'unknown'), ('ED_India_', 'unknown'), ('phone_conj_masks_2587219214', 'unknown'),
    ('', 'unknown'), (None, 'unknown'), (12, 'unknown')])
def test_country_strict_prefixes(image_id, country):
    assert SM.country_of_id(image_id) == country


def test_summary_rows_by_country():
    gt = box(0, 10, 0, 10)
    ids = ['ED_India_1', 'ED_India_2', 'ED_Italy_1', 'ED_Spain_1']
    recs = [SM.image_record('seg_ext', 'arm', 'eyes_defy', i, i, 0, 'ok', pred=gt, gt=gt, accepted=True) for i in ids]
    recs += [SM.image_record('seg_ext', 'arm', 'phone_conj_masks', f'phone_conj_masks_{k}', f'p{k}', 0, 'ok',
                             pred=gt, gt=gt, accepted=True) for k in range(3)]
    S = SM.summarize(SM.records_frame(recs))
    ed = S[S.dataset == 'eyes_defy'].set_index('country')
    assert list(ed.index) == ['all', 'India', 'Italy', 'unknown']
    assert (ed.loc['all', 'n_total'], ed.loc['India', 'n_total'], ed.loc['Italy', 'n_total'],
            ed.loc['unknown', 'n_total']) == (4, 2, 1, 1)
    assert list(S[S.dataset == 'phone_conj_masks'].country) == ['all']     # все unknown → только «all»


def test_csv_columns_and_no_paths():
    P = SM.records_frame(_records())
    assert list(P.columns) == list(SM.CSV_COLUMNS)
    for c in ('image', 'file', 'path', 'mask', 'mask_conj', 'hb', 'Hb', 'name'):
        assert c not in P.columns
    assert not P['id'].str.contains(r'[/\\]|\.(?:jpe?g|png|bmp)$', flags=re.IGNORECASE).any()
    assert isinstance(SM.summarize(P), pd.DataFrame)
