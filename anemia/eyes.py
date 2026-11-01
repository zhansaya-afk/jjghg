"""Глаза на обычном фото лица (2026-10-31): рамки глаз по точкам MediaPipe Face Landmarker, перевод координат
«вырезка ↔ исходное фото» и анатомические зоны по контуру глаза.

Точки лица — только «где глаз», а не доказательство видимой конъюнктивы: веко подтверждает сегментатор
(predict.learned_conj + anemia.anatomy), а зоны отсюда лишь запрещают маскам выходить за глаз.
Все функции чистые: координаты и маски на входе, числа и маски на выходе.

Координаты: рамка box = (x0, y0, x1, y1) — в пикселях исходного фото (после поворота по EXIF); вырезка —
rgb[y0:y1, x0:x1], уменьшенная до regions.WORK_SIDE с масштабом s = ширина вырезки / (x1 − x0).
"""
import cv2
import numpy as np

# Рамка глаза в ширинах глаза W (по уголкам глаза): по бокам SIDE, вверх от верхнего века UP, вниз от нижнего века
# DOWN. Вниз — с запасом: оттянутое веко с конъюнктивой уходит ниже контура глаза, который видит модель лица.
BOX_SIDE, BOX_UP, BOX_DOWN = 0.35, 0.35, 1.1
BOX_DOWN_RETRY = 1.6          # второй раз, если сегментатор нашёл веко у нижнего края рамки
CONJ_ZONE_UP = 0.25           # зона века: от нижнего контура глаза − 0,25 W (сетка лица может не дойти до века)
CONJ_ZONE_SIDE = 0.15         # по бокам ±0,15 W от уголков
SCLERA_ZONE_GROW = 0.12       # склера — внутри контура глазной щели, расширенного на 0,12 W (неточность точек)
ZONE_LOST_MAX = 0.5           # больше половины маски века вне зоны → глаз не принимается
MIN_EYE_PX = 60               # ширина глаза на исходном фото, px; меньше — цвет ткани не разобрать


def eye_width(poly):
    return float(np.ptp(np.asarray(poly)[:, 0]))


def eye_box(poly, shape, down=BOX_DOWN):
    """Контур глаза (точки N×2, пиксели фото) → рамка (x0, y0, x1, y1) в пределах фото."""
    p = np.asarray(poly, float)
    h, w = shape[:2]
    W = max(eye_width(p), 1.0)
    x0, x1 = p[:, 0].min() - BOX_SIDE * W, p[:, 0].max() + BOX_SIDE * W
    y0, y1 = p[:, 1].min() - BOX_UP * W, p[:, 1].max() + down * W
    return (int(max(0, np.floor(x0))), int(max(0, np.floor(y0))), int(min(w, np.ceil(x1))), int(min(h, np.ceil(y1))))


def crop_scale(box, crop_shape):
    """Масштаб вырезки относительно исходного фото (одинаковый по осям: _prepare сохраняет пропорции)."""
    return crop_shape[1] / (box[2] - box[0])


def to_crop(points, box, s):
    """Точки исходного фото → координаты вырезки."""
    p = np.asarray(points, float)
    return (p - np.array([box[0], box[1]], float)) * s


def to_photo(points, box, s):
    """Точки вырезки → координаты исходного фото."""
    p = np.asarray(points, float)
    return p / s + np.array([box[0], box[1]], float)


def paste_mask(mask, box, photo_shape):
    """Маска вырезки → маска размера исходного фото (вне рамки — False)."""
    out = np.zeros(photo_shape[:2], bool)
    x0, y0, x1, y1 = box
    if mask is None or x1 <= x0 or y1 <= y0:
        return out
    m = cv2.resize(np.asarray(mask, np.uint8), (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
    out[y0:y1, x0:x1] = m > 0
    return out


def lower_arc(poly):
    """Нижняя дуга контура глаза MediaPipe (первые 9 точек R_EYE / L_EYE: от уголка к уголку по нижнему веку)."""
    return np.asarray(poly, float)[:9]


def zones(crop_shape, poly_crop, iris_crop=None):
    """Разрешённые зоны на вырезке → (зона века, зона склеры), bool.
    Зона века — полоса под нижним контуром глаза: от контура − CONJ_ZONE_UP W до низа вырезки, по ширине глаза
    ± CONJ_ZONE_SIDE W, и не выше центра радужки. Зона склеры — контур глазной щели, расширенный на
    SCLERA_ZONE_GROW W, без круга радужки."""
    h, w = crop_shape[:2]
    p = np.asarray(poly_crop, float)
    W = max(eye_width(p), 1.0)
    low = lower_arc(p)
    low = low[np.argsort(low[:, 0])]
    xs = np.arange(w, dtype=float)
    y_low = np.interp(xs, low[:, 0], low[:, 1])
    yy = np.arange(h, dtype=float)[:, None]
    xin = (xs >= p[:, 0].min() - CONJ_ZONE_SIDE * W) & (xs <= p[:, 0].max() + CONJ_ZONE_SIDE * W)
    conj = (yy >= y_low[None, :] - CONJ_ZONE_UP * W) & xin[None, :]
    poly_mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(poly_mask, [np.round(p).astype(np.int32)], 1)
    k = max(1, round(SCLERA_ZONE_GROW * W))
    sclera = cv2.dilate(poly_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))) > 0
    if iris_crop is not None:
        cx, cy, r = iris_crop
        yg, xg = np.ogrid[0:h, 0:w]
        sclera &= np.hypot(xg - cx, yg - cy) > 0.95 * r
        conj &= np.broadcast_to(yy >= cy, conj.shape)
    return conj, sclera


def constrain(conj, sclera, conj_zone, sclera_zone):
    """Маски вне зон глаза убираются → (conj, sclera, conj_ok, доля века вне зоны, доля склеры вне зоны).
    conj_ok False — больше ZONE_LOST_MAX маски века вне зоны: веко не там, где глаз."""
    conj = np.asarray(conj, bool)
    sclera = np.asarray(sclera, bool)
    lost_c = 1 - (conj & conj_zone).sum() / conj.sum() if conj.any() else 0.0
    lost_s = 1 - (sclera & sclera_zone).sum() / sclera.sum() if sclera.any() else 0.0
    return conj & conj_zone, sclera & sclera_zone, lost_c <= ZONE_LOST_MAX, float(lost_c), float(lost_s)
