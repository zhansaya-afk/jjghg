"""Анатомические ограничения масок (anemia/anatomy.py) и их место в анализе фото. Синтетика — проверка кода,
не точности."""
import cv2
import numpy as np
import pytest

from anemia import anatomy as AN
from anemia import predict as PR
from anemia import segment_baseline as SB

H, W = 320, 240


def strip(h=H, w=W, x0=30, x1=210, y=200, t=24):
    """Полоса века: дуга шириной x1−x0, толщиной t."""
    m = np.zeros((h, w), np.uint8)
    cv2.ellipse(m, ((x0 + x1) // 2, y), ((x1 - x0) // 2, t), 0, 0, 180, 1, -1)
    cv2.ellipse(m, ((x0 + x1) // 2, y), ((x1 - x0) // 2 - 6, t // 3), 0, 0, 180, 0, -1)
    return m.astype(bool)


def blob(cx, cy, r, h=H, w=W):
    m = np.zeros((h, w), np.uint8)
    cv2.circle(m, (cx, cy), r, 1, -1)
    return m.astype(bool)


def test_finger_blob_removed_strip_kept():
    s = strip()
    finger = blob(200, 300, 12)
    r = AN.check(s | finger, None)
    assert r['conj_ok'], r['conj_why']
    assert not (r['conj'] & finger).any()
    assert (r['conj'] == s).all()
    assert r['stats']['убрано частей вне полосы века, px'] == int(finger.sum())


def test_fragmented_mask_refused():
    m = blob(40, 120, 30) | blob(120, 200, 30) | blob(200, 280, 30)   # три равные части: какая веко — неизвестно
    r = AN.check(m, None)
    assert not r['conj_ok']
    assert 'раздроблена' in r['conj_why']


def test_narrow_strip_refused():
    r = AN.check(strip(x0=100, x1=170), None)           # 29% ширины кадра < 42%
    assert not r['conj_ok']
    assert 'ширины кадра' in r['conj_why']


def test_round_spot_refused():
    r = AN.check(blob(120, 200, 60), None)              # пятно 120×120: высота/ширина 1 > 1,02? — нет; круг ещё выше
    m = np.zeros((H, W), bool)
    m[60:300, 60:180] = True                            # высота/ширина 2
    r = AN.check(m, None)
    assert not r['conj_ok']
    assert 'пятно' in r['conj_why']


def test_sclera_above_strip_kept_below_and_side_removed():
    s = strip()
    above = blob(120, 150, 20)
    below = np.zeros((H, W), bool)
    below[300:306, 80:160] = True                        # «склера» под веком (насадка, кожа)
    side = blob(200, 12, 6)                              # далеко над глазом (бровь, лоб)
    r = AN.check(s, above | below | side)
    assert r['sclera_ok'], r['sclera_why']
    assert (r['sclera'] == above).all()


def test_sclera_mostly_outside_dropped():
    s = strip()
    below = np.zeros((H, W), bool)
    below[290:318, 20:220] = True
    r = AN.check(s, blob(120, 160, 8) | below)
    assert not r['sclera_ok']
    assert not r['sclera'].any()
    assert 'склера' in r['sclera_why']


def test_empty_conj():
    r = AN.check(np.zeros((H, W), bool), blob(100, 100, 10))
    assert not r['conj_ok'] and not r['sclera_ok']


def test_fit_limits_uses_only_given_masks_and_caps_sclera_below():
    stats = [AN.image_stats(strip(x0=20 + i, x1=220 - i), blob(120, 150 + 30 * (i % 2), 15)) for i in range(20)]
    lim = AN.fit_limits(stats)
    assert lim['sclera_cy'][1] <= 0.0                     # центр склеры не ниже центра полосы — анатомия
    assert 0.3 < lim['wfrac_min'] < 0.5
    assert lim == AN.fit_limits(list(stats) + [None])     # снимки без маски врача не влияют
    with pytest.raises(ValueError):
        AN.fit_limits([None])


class MaskSeg:
    """Сегментатор-заглушка: вероятность века = заданная маска (по признакам положения xx/w, yy/h)."""

    def __init__(self, mask):
        self.mask = mask.astype(np.float32)

    def predict_proba(self, X):
        h, w = self.mask.shape
        xs = np.clip((X[:, -4] * w).astype(int), 0, w - 1)
        ys = np.clip((X[:, -3] * h).astype(int), 0, h - 1)
        p = self.mask[ys, xs]
        return np.stack([1 - p, p], 1)


def fake_det(sclera=None):
    rng = np.random.default_rng(0)
    crop = np.clip(0.5 + 0.05 * rng.standard_normal((H, W, 3)), 0, 1).astype(np.float32)
    sclera = np.zeros((H, W), bool) if sclera is None else sclera
    return {'crop': crop, 'conj': np.zeros((H, W), bool), 'sclera': sclera, 'iris': None,
            'use': {'conj', 'sclera'} if sclera.any() else {'conj'}, 'kind': 'глаз крупным планом', 'notes': [],
            'src_side': 1000, 'score': 3.0}


def seg_for(mask):
    small = SB.resize_to(mask, (SB.SEG_SIZE, round(SB.SEG_SIZE * W / H)), nearest=True)
    return MaskSeg(small)


def test_learned_conj_refuses_fragmented_lid_and_never_marks_fit():
    m = blob(60, 120, 30) | blob(180, 260, 32)
    d = PR.learned_conj(fake_det(), seg_for(m))
    assert d['use'] == set()
    assert any('анатомическая проверка века не пройдена' in n for n in d['notes'])
    ts = PR.tissue_status(d['crop'], d)
    assert ts['conj'] != 'пригодна' and ts['sclera'] != 'пригодна'


def test_learned_conj_without_anatomy_keeps_old_behaviour():
    m = blob(60, 120, 30) | blob(180, 260, 32)
    d = PR.learned_conj(fake_det(), seg_for(m), limits=None)
    assert 'conj' in d['use'] and d['conj'].sum() > 0.9 * m.sum()


def test_learned_conj_drops_sclera_below_lid():
    s = strip()
    below = np.zeros((H, W), bool)
    below[290:318, 20:220] = True
    d = PR.learned_conj(fake_det(sclera=below), seg_for(s | blob(220, 300, 8)))
    assert 'sclera' not in d['use']
    assert not d['sclera'].any()
    assert d.get('anatomy')


def test_analyze_photo_refuses_without_risk_and_hb(monkeypatch):
    m = blob(60, 120, 30) | blob(180, 260, 32)
    det = fake_det()
    monkeypatch.setattr(PR, '_detect', lambda rgb, lm=None: dict(det, eye='глаз крупным планом'))
    bundle = {'models': {'eyes_defy': {'title': 'тест'}}, 'conj_seg': seg_for(m)}
    rgb = np.random.default_rng(1).uniform(0.2, 0.9, (600, 450, 3)).astype(np.float32)   # цветной кадр
    r = PR.analyze_photo(rgb, sex='F', age=30, bundle=bundle, landmarker=None)
    assert r['status'] != 'ok'
    assert r['risk'] is None and r['hb_range'] is None and r['p_anemia'] is None
    assert 'пригодна' not in r['tissue_status'].values()
    assert any('анатомия масок' in x for x in r['diag'])


def test_overlay_draws_exactly_mask_boundaries():
    s = strip()
    sc = blob(120, 150, 20)
    r = {'crop': np.full((H, W, 3), 0.5, np.float32), 'conj': s, 'sclera': sc}
    vis = PR.overlay(r)
    base = (np.clip(r['crop'], 0, 1) * 255).astype(np.uint8)
    changed = (vis != base).any(-1)
    for m in (s, sc):
        edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        near = cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        assert changed[edge].mean() == 1.0                  # каждый пиксель края маски нарисован
        assert (changed & near).sum() > 0
    border = np.zeros_like(changed)
    for m in (s, sc):
        edge = m & ~cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        border |= cv2.dilate(edge.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert not (changed & ~border).any()                    # ничего не нарисовано вне краёв масок
