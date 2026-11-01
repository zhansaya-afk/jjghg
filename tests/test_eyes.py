"""Маршрут «фото лица → глаза → вырезка → сегментация → анатомия» (anemia/eyes.py, predict.find_tissues):
геометрия рамки, перевод координат вырезка ↔ фото, зоны глаза, ложное «лицо», каждый глаз отдельно и сохранение
прежнего маршрута макроснимка. Синтетика и заглушки детектора/сегментатора — проверка кода, не точности."""
from collections import defaultdict
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from anemia import eyes as EYES
from anemia import predict as PR
from anemia import regions as R

PH, PW = 700, 900          # фото лица
EW = 120                   # ширина глаза на фото, px


def eye_poly(cx, cy, w=EW):
    """Контур глаза в порядке MediaPipe R_EYE / L_EYE: 9 точек нижней дуги от уголка к уголку, затем 7 верхних."""
    a, b = w / 2, 0.3 * w
    low = [(cx + a * np.cos(t), cy + b * np.sin(t)) for t in np.linspace(np.pi, 0, 9)]
    up = [(cx + a * np.cos(t), cy - b * np.sin(t)) for t in np.linspace(0, np.pi, 9)[1:-1]]
    return np.array(low + up, float)


EYES_AT = {'правый': (300, 300), 'левый': (600, 300)}     # правый глаз человека — слева на фото


def lid_band(cx, cy, w=EW, shape=(PH, PW)):
    """Вывернутое нижнее веко: красная дуга под нижним контуром глаза."""
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]].astype(float)
    a, b = w / 2, 0.3 * w
    inside = np.abs(xx - cx) < 0.45 * w
    y_arc = cy + b * np.sqrt(np.clip(1 - ((xx - cx) / a) ** 2, 0, 1))
    return inside & (yy > y_arc + 4) & (yy < y_arc + 30)


def face_photo(lids=('правый', 'левый'), draw_eyes=True, seed=0):
    """Синтетическое «лицо»: кожа, у глаз — белая склера и тёмная радужка; у глаз из lids — вывернутое веко."""
    rng = np.random.default_rng(seed)
    img = np.empty((PH, PW, 3), np.float32)
    img[:] = (0.80, 0.62, 0.52)
    img += 0.015 * rng.standard_normal(img.shape).astype(np.float32)
    truth = {}
    for side, (cx, cy) in EYES_AT.items():
        if draw_eyes:
            m = np.zeros((PH, PW), np.uint8)
            cv2.fillPoly(m, [np.round(eye_poly(cx, cy)).astype(np.int32)], 1)
            img[m > 0] = (0.93, 0.92, 0.90)
            cv2.circle(img, (cx, cy), 22, (0.18, 0.12, 0.08), -1)
        band = lid_band(cx, cy) if side in lids else np.zeros((PH, PW), bool)
        img[band] = (0.78, 0.22, 0.26)
        truth[side] = band
    return np.clip(img, 0, 1), truth


def landmarks(sides=EYES_AT, iris_r=22):
    """478 точек лица (доли кадра) с контурами и радужками глаз sides; остальные точки — в центре."""
    P = np.tile([[PW / 2, PH / 2]], (478, 1)).astype(float)
    for side, (cx, cy) in sides.items():
        eye, iris = (R.R_EYE, R.R_IRIS) if side == 'правый' else (R.L_EYE, R.L_IRIS)
        P[eye] = eye_poly(cx, cy)
        P[iris] = [(cx, cy), (cx + iris_r, cy), (cx, cy - iris_r), (cx - iris_r, cy), (cx, cy + iris_r)]
    return [SimpleNamespace(x=x / PW, y=y / PH) for x, y in P]


class FakeLandmarker:
    """Детектор лица — заглушка: точки в долях кадра, как у MediaPipe (масштаб кадра не важен)."""

    def __init__(self, pts=None):
        self.pts = pts

    def detect(self, img):
        return SimpleNamespace(face_landmarks=[self.pts] if self.pts is not None else [])


class RedSeg:
    """Сегментатор века — заглушка: веко = заметно красные пиксели (Lab a* после выравнивания света)."""

    def predict_proba(self, X):
        p = (X[:, 5] > 0.6).astype(np.float32)          # столбец 5 — a*/50 без размытия (segment_baseline)
        return np.stack([1 - p, p], 1)


# ---------------------------------------------------------------- геометрия и координаты
def test_eye_box_covers_eye_with_margins_and_stays_in_frame():
    p = eye_poly(300, 300)
    x0, y0, x1, y1 = EYES.eye_box(p, (PH, PW))
    assert x0 <= p[:, 0].min() - EYES.BOX_SIDE * EW + 1 and x1 >= p[:, 0].max() + EYES.BOX_SIDE * EW - 1
    assert y0 <= p[:, 1].min() - EYES.BOX_UP * EW + 1 and y1 >= p[:, 1].max() + EYES.BOX_DOWN * EW - 1
    assert EYES.eye_box(p, (PH, PW), EYES.BOX_DOWN_RETRY)[3] > y1          # повтор — рамка ниже
    edge = EYES.eye_box(eye_poly(40, 650), (PH, PW))                       # у края фото — обрезается кадром
    assert edge[0] == 0 and edge[3] == PH and edge[2] <= PW


def test_crop_photo_coordinates_round_trip():
    box = (213, 155, 401, 449)
    crop_shape = (300, 192)
    s = EYES.crop_scale(box, crop_shape)
    pts = np.array([[213, 155], [300.5, 222.25], [400, 448]], float)
    back = EYES.to_photo(EYES.to_crop(pts, box, s), box, s)
    assert np.abs(back - pts).max() < 1e-9
    assert np.allclose(EYES.to_crop([[box[0], box[1]]], box, s), 0)


def test_paste_mask_puts_crop_mask_back_where_it_was_on_photo():
    rgb, truth = face_photo()
    box = EYES.eye_box(eye_poly(*EYES_AT['левый']), rgb.shape)
    x0, y0, x1, y1 = box
    crop = R._prepare(rgb[y0:y1, x0:x1])                  # как в маршруте: вырезка приведена к WORK_SIDE
    s = EYES.crop_scale(box, crop.shape)
    assert abs(crop.shape[0] / (y1 - y0) - s) < 0.01        # масштаб одинаков по осям (пропорции сохранены)
    red = (crop[..., 0] - crop[..., 1]) > 0.35            # веко на вырезке
    back = EYES.paste_mask(red, box, rgb.shape)
    t = truth['левый']
    iou = (back & t).sum() / (back | t).sum()
    assert iou > 0.9
    cy_t, cx_t = np.argwhere(t).mean(0)
    cy_b, cx_b = np.argwhere(back).mean(0)
    assert abs(cx_t - cx_b) <= 1 and abs(cy_t - cy_b) <= 1   # центр маски на фото — с точностью ≤ 1 px
    assert not back[:y0].any() and not back[:, :x0].any()


def test_zones_lid_below_eye_sclera_without_iris():
    shape = (300, 260)
    poly = eye_poly(130, 100, 160)
    iris = (130, 100, 25)
    cz, sz = EYES.zones(shape, poly, iris)
    assert cz[200, 130] and not cz[60, 130]               # под глазом можно, над глазом нельзя
    assert not cz[150, 5]                                 # далеко сбоку от уголка нельзя
    assert not cz[: int(iris[1])].any()                   # не выше центра радужки
    assert sz[100, 70] and not sz[100, 130] and not sz[250, 130]   # склера: в щели, не на радужке и не под веком


def test_constrain_drops_outside_parts_and_refuses_mostly_outside():
    zone = np.zeros((100, 100), bool)
    zone[50:] = True
    conj = np.zeros_like(zone)
    conj[60:70, 10:90] = True
    conj[5:10, 10:20] = True                              # «веко» над глазом
    c, _, ok, lost, _ = EYES.constrain(conj, np.zeros_like(zone), zone, zone)
    assert ok and not c[:50].any() and c[60:70, 10:90].all() and 0 < lost < 0.1
    conj[0:40, :] = True
    _, _, ok, lost, _ = EYES.constrain(conj, np.zeros_like(zone), zone, zone)
    assert not ok and lost > EYES.ZONE_LOST_MAX


# ---------------------------------------------------------------- маршрут лица
def test_face_route_both_eyes_judged_and_mask_lands_on_lid():
    rgb, truth = face_photo()
    _, det, diag = PR.find_tissues(rgb, RedSeg(), FakeLandmarker(landmarks()))
    sides = {e['side']: e for e in det['eyes']}
    assert set(sides) == {'правый глаз', 'левый глаз'}
    assert all(e['accepted'] for e in sides.values())
    assert det['eye'].startswith('лицо в кадре')
    side = det['eye_side'].split()[0]
    on_photo = EYES.paste_mask(det['conj'], det['eye_box'], rgb.shape)
    assert on_photo.any()
    assert (on_photo & truth[side]).sum() / on_photo.sum() > 0.8        # маска — на веке этого глаза, не на коже
    other = 'левый' if side == 'правый' else 'правый'
    assert not (on_photo & truth[other]).any()
    assert sum('рамка x' in x for x in diag) >= 2


def test_each_eye_separately_only_everted_eye_accepted():
    rgb, truth = face_photo(lids=('левый',))
    _, det, _ = PR.find_tissues(rgb, RedSeg(), FakeLandmarker(landmarks()))
    sides = {e['side']: e for e in det['eyes']}
    assert not sides['правый глаз']['accepted'] and sides['левый глаз']['accepted']
    assert det['eye_side'] == 'левый глаз'
    on_photo = EYES.paste_mask(det['conj'], det['eye_box'], rgb.shape)
    assert (on_photo & truth['левый']).sum() / on_photo.sum() > 0.8


def test_landmarks_alone_do_not_prove_conjunctiva():
    rgb, _ = face_photo(lids=())                          # глаза есть, веко не вывернуто
    bundle = {'models': {'eyes_defy': {'title': 'тест'}}, 'conj_seg': RedSeg()}
    r = PR.analyze_photo(rgb, sex='F', age=30, bundle=bundle, landmarker=FakeLandmarker(landmarks()))
    assert r['eyes'] and not any(e['accepted'] for e in r['eyes'])
    assert r['status'] != 'ok'
    assert r['risk'] is None and r['p_anemia'] is None and r['hb_range'] is None
    assert any('прежний маршрут' in x for x in r['diag'])


def test_lid_outside_eye_zone_refused():
    rgb, _ = face_photo(lids=())
    cx, cy = EYES_AT['левый']
    patch = np.zeros((PH, PW), bool)
    patch[cy - 70:cy - 45, cx - 55:cx + 55] = True        # красная полоса над глазом (бровь/веко сверху)
    rgb[patch] = (0.78, 0.22, 0.26)
    e = next(z for z in PR.face_eyes(rgb, FakeLandmarker(landmarks())) if z['side'] == 'левый глаз')
    z = PR.analyze_eye(rgb, e, RedSeg())
    assert 'conj' not in z['use']


def test_false_face_without_iris_contrast_falls_back_to_old_route(monkeypatch):
    rgb, _ = face_photo(lids=('правый', 'левый'), draw_eyes=False)   # «точки лица» на коже: радужки нет
    calls = []
    real = PR._closeup_route
    monkeypatch.setattr(PR, '_closeup_route', lambda *a, **k: calls.append(1) or real(*a, **k))
    _, det, _ = PR.find_tissues(rgb, RedSeg(), FakeLandmarker(landmarks()))
    assert calls, 'нет отката на прежний маршрут'
    assert det['eyes'] and not any(e['accepted'] for e in det['eyes'])
    assert all('ошибочно' in e['why'] for e in det['eyes'])
    assert 'eye_side' not in det


def test_small_eyes_skipped():
    e = {'side': 'правый глаз', 'poly': eye_poly(300, 300, 40), 'iris': (300, 300, 8), 'W': 40.0,
         'box': EYES.eye_box(eye_poly(300, 300, 40), (PH, PW))}
    z = PR.analyze_eye(face_photo()[0], e, RedSeg())
    assert z['use'] == set() and 'мелкий' in z['eye_log'][-1]


# ---------------------------------------------------------------- прежний маршрут не меняется
@pytest.mark.parametrize('lm', [None, FakeLandmarker(None)])
def test_no_face_gives_exactly_old_route(monkeypatch, lm):
    rgb = face_photo()[0][150:550, 150:750].copy()
    old = PR._closeup_route(rgb, RedSeg(), lm)
    new = PR.find_tissues(rgb, RedSeg(), lm)
    assert new[1]['eyes'] == []
    for k in ('conj', 'sclera'):
        assert np.array_equal(old[1][k], new[1][k])
    assert old[1]['use'] == new[1]['use'] and old[2] == new[2]


def test_overlay_photo_boxes_and_text_per_eye(monkeypatch):
    rgb, _ = face_photo(lids=('левый',))
    bundle = {'models': {'eyes_defy': {'title': 'тест'}}, 'conj_seg': RedSeg()}
    monkeypatch.setattr(PR, 'ADULT_MODEL', defaultdict(lambda: 'нет модели'))   # модель риска не нужна: стоп до неё
    r = PR.analyze_photo(rgb, sex='F', age=30, bundle=bundle, landmarker=FakeLandmarker(landmarks()))
    vis = PR.overlay_photo(r)
    assert vis is not None and vis.shape[:2] == r['photo_preview'].shape[:2]
    k = r['preview_scale']
    acc = {e['side']: e for e in r['eyes']}
    x0, y0 = (round(v * k) for v in acc['левый глаз']['box'][:2])
    assert tuple(vis[y0, x0 + 5]) == (0, 200, 0)                     # принятый глаз — зелёная рамка
    x0, y0 = (round(v * k) for v in acc['правый глаз']['box'][:2])
    assert tuple(vis[y0, x0 + 5]) == (220, 0, 0)                     # отказ — красная
    txt = PR.format_result(r)
    assert 'правый глаз: отказ' in txt and 'левый глаз: принят' in txt
    assert PR.overlay_photo({'photo_preview': None}) is None
