"""Поиск тканей глаза на снимке: конъюнктива нижнего века и склера, три режима работы.

Режимы (параметр ``mode``):
    'auto'   — система сама решает, что на фото (вырезанная ткань, крупный план века, глаз целиком, лицо);
    'conj'   — на фото только конъюнктива (вывернутое веко крупно или уже вырезанная ткань): склеру не ищем;
    'sclera' — на фото только склера (белок глаза): конъюнктиву не ищем;
    'both'   — глаз с оттянутым веком: ищем обе ткани; если одна не нашлась, работаем по другой.

Почему не пороги «как в v11»: прошлый сегментатор учился только на кадрах Eyes-defy (насадка, весь кадр — глаз)
и опирался на положение относительно радужки. На крупном плане века радужки нет, признаки положения теряют смысл —
маска пустая, и система отвечала «переснять». Здесь каждая ткань ищется по собственным свойствам:
    склера — самая светлая и бесцветная ткань рядом с радужкой (низкая насыщенность C* в CIELAB);
    конъюнктива — самая «красно-розовая» ткань (высокий a*, меньший угол оттенка h_ab, чем у кожи) под радужкой
                  и склерой; границу уточняет GrabCut (OpenCV) по затравкам из этих правил.
Каждая найденная маска проходит проверку правдоподобия (цвет, размер, положение). Не прошла — ткань считается
не найденной, и снимок оценивается по другой ткани, а не по коже или блику.

Соглашение: изображения — float32 RGB [0, 1] (sRGB), маски — bool H×W.
"""
import cv2
import numpy as np

from .color import lin_to_srgb, srgb_to_lin

WORK_SIDE = 320          # длинная сторона вырезки, на которой ищутся ткани
MODES = ('auto', 'conj', 'sclera', 'both')
MIN_CONJ_FRAC = 0.01     # минимум площади вырезки для конъюнктивы
MIN_SCLERA_FRAC = 0.006  # и для склеры
CUTOUT_BORDER_MIN = 0.5  # фон вырезанной ткани занимает ≥ 50% края кадра (CP-AnemiC: от 74%)

# MediaPipe Face Landmarker: контуры глаз и радужки (478 точек, радужка 468–477)
R_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
L_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
R_IRIS, L_IRIS = [468, 469, 470, 471, 472], [473, 474, 475, 476, 477]


# ------------------------------------------------------------------ цветовые карты
def gray_world(rgb):
    """«Серый мир» (shades of gray, p=6) + выравнивание яркости. Только для поиска тканей, не для признаков Hb."""
    lin = srgb_to_lin(np.clip(rgb, 0, 1)).reshape(-1, 3)
    g = (np.mean(lin ** 6, axis=0) + 1e-12) ** (1 / 6)
    lin = lin * (g.mean() / g)
    lin = lin * (0.8 / max(float(np.percentile(lin.max(1), 95)), 1e-3))
    return lin_to_srgb(lin).reshape(rgb.shape)


def color_maps(rgb):
    """Карты CIELAB по нормированному свету: L*, a*, b*, насыщенность C*, угол оттенка h_ab (градусы)."""
    n = gray_world(rgb)
    lab = cv2.cvtColor(np.ascontiguousarray(n, np.float32), cv2.COLOR_RGB2Lab)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]
    raw = cv2.cvtColor(np.ascontiguousarray(np.clip(rgb, 0, 1), np.float32), cv2.COLOR_RGB2Lab)
    # «белый лист»: самые светлые пиксели кадра (обычно склера) считаем белыми. Серый мир делает
    # нейтральной кожу, и на светлой коже склера кажется цветной — поэтому бесцветность берём по обоим
    lin = srgb_to_lin(np.clip(rgb, 0, 1)).reshape(-1, 3)
    wp = np.percentile(lin, 97, axis=0) + 1e-6
    wl = cv2.cvtColor(np.ascontiguousarray(lin_to_srgb(lin * (wp.mean() / wp)).reshape(rgb.shape), np.float32),
                      cv2.COLOR_RGB2Lab)
    C_white = np.minimum(np.hypot(a, b), np.hypot(wl[..., 1], wl[..., 2]))
    return {'C_white': C_white,'n': n, 'L': L, 'a': a, 'b': b, 'C': np.hypot(a, b), 'hab': np.degrees(np.arctan2(b, a)),
            'L_raw': raw[..., 0], 'a_raw': raw[..., 1], 'C_raw': np.hypot(raw[..., 1], raw[..., 2]),
            'v': np.clip(rgb, 0, 1).max(-1)}


def redness(maps):
    """Насколько пиксель похож на слизистую: a* (красный) минус часть b* (жёлтый, у кожи больше)."""
    return maps['a'] - 0.45 * np.maximum(maps['b'], 0)


def _otsu(x, mask=None):
    v = x[mask] if mask is not None else x.ravel()
    if v.size < 20:
        return float(np.median(x))
    lo, hi = np.percentile(v, [1, 99])
    if hi - lo < 1e-6:
        return float(lo)
    u8 = (np.clip((v - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    t, _ = cv2.threshold(u8.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(lo + t / 255 * (hi - lo))


def _components(mask, min_area, keep_frac=0.25, max_parts=3):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return np.zeros_like(mask, bool)
    areas = st[1:, cv2.CC_STAT_AREA]
    order = np.argsort(-areas)
    keep = [1 + i for i in order[:max_parts] if areas[i] >= max(min_area, keep_frac * areas.max())]
    return np.isin(lab, keep)


def _fill_holes(mask):
    m = mask.astype(np.uint8)
    h, w = m.shape
    flood = np.pad(m, 1).copy()
    cv2.floodFill(flood, None, (0, 0), 1)
    return mask | (flood[1:-1, 1:-1] == 0)


def _smooth(mask, k=5):
    m = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0


# ------------------------------------------------------------------ фон «уже вырезанной» ткани
def dark_border(rgb):
    """Почти чёрные пиксели, связанные с краем кадра: фон за вырезанной тканью, волосы, тёмный фон за головой.
    -> bool-маска (пустая, если такого нет). У снимков Eyes-defy такого фона нет (≤ 0,4% кадра)."""
    v = rgb.max(-1)
    n, lab = cv2.connectedComponents((v < 0.05).astype(np.uint8), 8)
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    return np.isin(lab, border[border > 0])


def cutout_background(rgb):
    """Фон снимка, где ткань уже вырезана (как в CP-AnemiC): почти чёрные пиксели, связанные с краем кадра.
    -> маска фона (bool) или None, если такого фона нет.
    Вырезанную ткань фон окружает: он занимает большую часть края кадра (у CP-AnemiC ≥ 74%). Тёмная полоса
    только у одного края (волосы, тень, тёмная одежда на обычном фото глаза) — не фон вырезки."""
    bg = dark_border(rgb)
    edge = np.concatenate([bg[0], bg[-1], bg[:, 0], bg[:, -1]])
    return bg if 0.12 < bg.mean() < 0.9 and edge.mean() >= CUTOUT_BORDER_MIN else None


# ------------------------------------------------------------------ радужка
def _iris_score(L, C, cx, cy, r):
    """Контраст «тёмный круг на светлом»: средняя L* кольца снаружи минус L* внутри круга.
    Кольцо — по бокам и снизу (сверху радужку часто закрывает верхнее веко)."""
    h, w = L.shape
    yy, xx = np.ogrid[0:h, 0:w]
    d = np.hypot(xx - cx, yy - cy)
    inside = d < 0.8 * r
    ring = (d > 1.15 * r) & (d < 1.6 * r) & (yy > cy - 0.5 * r)
    if inside.sum() < 20 or ring.sum() < 20:
        return -1.0
    return float(np.median(L[ring]) - np.median(L[inside]))


def find_iris(rgb, maps=None, min_r=0.07, max_r=0.32, min_contrast=10, with_score=False):
    """Радужка: тёмный круг, окружённый более светлой тканью (склера, веко) -> (cx, cy, r) или None.
    Кандидаты — окружности Хафа и тёмные округлые пятна; выбирается самый контрастный круг.
    min_r / max_r — радиус в долях короткой стороны (по умолчанию — глаз крупным планом, как при обучении);
    with_score=True -> ((cx, cy, r), контраст) — насколько уверенно определена граница радужки."""
    maps = maps or color_maps(rgb)
    L, C = cv2.GaussianBlur(maps['L'], (0, 0), 1.5), maps['C']
    h, w = L.shape
    side = min(h, w)
    cands = []
    u8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    for p2 in (30, 22, 16):
        circ = cv2.HoughCircles(u8, cv2.HOUGH_GRADIENT, dp=1.5, minDist=side / 6, param1=60, param2=p2,
                                minRadius=int(min_r * side), maxRadius=int(max_r * side))
        if circ is not None:
            cands += [tuple(c) for c in circ[0][:8]]
            break
    for q in (5, 10, 16):
        dark = cv2.morphologyEx((L < np.percentile(L, q)).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, st, cen = cv2.connectedComponentsWithStats(dark, 8)
        for i in range(1, n):
            x, y, bw, bh, a = st[i]
            if a < 0.008 * h * w * (min_r / 0.07) ** 2 or bw > 0.6 * w or bh > 0.7 * h:
                continue
            if a / (bw * bh) > 0.45 and min(bw, bh) / max(bw, bh) > 0.45:
                cands.append((cen[i][0], cen[i][1], max(bw, bh) / 2))
    best, best_s = None, 0.0
    for cx, cy, r in cands:
        if not (0.6 * r < cx < w - 0.6 * r and 0.3 * r < cy < h - 0.6 * r):
            continue                                   # обрезанный краем кадра круг — не радужка
        con = _iris_score(L, C, cx, cy, r)
        central = 1 - min(1, np.hypot((cx - w / 2) / w, (cy - h / 2) / h) * 1.2)
        s = con * (0.5 + central)
        if con > min_contrast and s > best_s:
            best, best_s, best_con = (float(cx), float(cy), float(r)), s, float(con)
    if with_score:
        return best, (best_con if best else 0.0)
    return best


# ------------------------------------------------------------------ склера
def find_sclera(rgb, maps=None, iris=None, exclude=None):
    """Склера: светлые и самые бесцветные (низкий C*) участки, примыкающие к радужке слева/справа/снизу.
    Без радужки — крупные бесцветные светлые области не у края кадра."""
    maps = maps or color_maps(rgb)
    L, C, v = maps['L'], maps['C_white'], maps['v']
    h, w = L.shape
    glare = (v > 0.97) & (C < 6)
    pool = ~glare if exclude is None else ~glare & ~exclude
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    if iris is not None:
        cx, cy, r = iris
        # открытая склера — по бокам радужки и (при оттянутом веке) под ней, не выше верхнего края радужки
        # форма открытого глаза — миндаль: сверху парабола (верхнее веко опускается к уголкам),
        # снизу эллипс (при оттянутом веке склера видна и под радужкой)
        u = (xx - cx) / (2.9 * r)
        upper = (yy - cy) > -1.0 * r * (1 - u ** 2)
        lower = np.hypot(u, (yy - cy) / (1.9 * r)) < 1
        near = upper & lower & (np.abs(u) < 1) & (np.hypot(xx - cx, yy - cy) > 0.85 * r)
        region = near & pool
    else:
        region = pool
    if region.sum() < 50:
        return np.zeros((h, w), bool)
    # бесцветность и светлота относительно самого кадра (не абсолютные пороги)
    c_thr = float(np.clip(np.percentile(C[region], 35), 5, 16))
    # порог светлоты — Оцу (склера против кожи/века), но не выше 60/70-го перцентиля: если склера занимает
    # почти всю зону, перцентиль попал бы внутрь неё и разрезал её пополам
    l_thr = float(max(40, min(_otsu(L, region), np.percentile(L[region], 60 if iris is not None else 70))))
    cand = region & (C < c_thr) & (L > l_thr)
    cand = _smooth(cand, 5)
    min_area = MIN_SCLERA_FRAC * h * w
    n, lab, st, cen = cv2.connectedComponentsWithStats(cand.astype(np.uint8), 8)
    frame = np.zeros((h, w), bool)
    frame[:2], frame[-2:], frame[:, :2], frame[:, -2:] = True, True, True, True
    keep = []
    for i in range(1, n):
        a = st[i, cv2.CC_STAT_AREA]
        if a < min_area:
            continue
        comp = lab == i
        if iris is None and (comp & frame).sum() > 0.15 * st[i, cv2.CC_STAT_WIDTH] + 0.15 * st[i, cv2.CC_STAT_HEIGHT]:
            continue                                    # стена/одежда/фон у края кадра
        if iris is not None:
            touch = cv2.dilate(comp.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
            d = np.hypot(xx[touch] - cx, yy[touch] - cy).min()
            if d > 1.35 * r:                            # склера прилегает к радужке
                continue
        keep.append((a * (1 + (c_thr - C[comp].mean()) / c_thr), i))
    if not keep:
        return np.zeros((h, w), bool)
    keep.sort(reverse=True)
    m = np.isin(lab, [i for s, i in keep[:2] if s >= 0.2 * keep[0][0]])
    return cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0


# ------------------------------------------------------------------ конъюнктива
def _upper_edge(h, w, iris, sclera):
    """Для каждого столбца — нижний край склеры/радужки (под ним начинается вывернутое веко).
    Где склеры нет, край интерполируется от соседних столбцов. -> массив длины w или None."""
    top = np.full(w, np.nan, np.float32)
    if sclera is not None and sclera.any():
        ys, xs = np.where(sclera)
        t = np.full(w, -1.0, np.float32)
        np.maximum.at(t, xs, ys.astype(np.float32))
        top = np.where(t >= 0, t, np.nan)
    if iris is not None:
        cx, cy, r = iris
        xs = np.arange(w)
        d = r ** 2 - (xs - cx) ** 2
        ib = np.where(d > 0, cy + np.sqrt(np.maximum(d, 0)), np.nan)
        top = np.where(np.isnan(top), ib, np.fmax(top, ib))
    ok = ~np.isnan(top)
    if ok.sum() < 3:
        return None
    xs = np.arange(w)
    return np.interp(xs, xs[ok], cv2.GaussianBlur(top[ok].reshape(1, -1), (0, 0), 3).ravel()
                     if ok.sum() > 7 else top[ok])


def find_conj(rgb, maps=None, iris=None, sclera=None, region=None, geometry=True):
    """Конъюнктива: связная «красно-розовая» ткань (высокий a*, кожа желтее) под радужкой/склерой.
    Гистерезис: затравки — самые красные пиксели зоны поиска, рост — в пиксели краснее порога Оцу,
    затем GrabCut уточняет край только в узкой полосе вокруг найденного. region — где разрешено искать.
    geometry=False — радужка и склера только исключаются, но не задают, где искать (на случай, если они найдены
    ошибочно, например на крупном плане века)."""
    maps = maps or color_maps(rgb)
    h, w = maps['L'].shape
    R = cv2.GaussianBlur(redness(maps), (0, 0), 2.0)
    v = maps['v']
    allowed = (v > 0.06) & ~((v > 0.97) & (maps['C'] < 6))
    if region is not None:
        allowed &= region
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    search = allowed.copy()
    if iris is not None:
        cx, cy, r = iris
        if geometry:
            search &= (yy > cy + 0.2 * r) & (yy < cy + 4.0 * r) & (np.abs(xx - cx) < 3.8 * r)
        allowed &= ~(np.hypot(xx - cx, yy - cy) < 1.05 * r)
        search &= allowed
    if sclera is not None and sclera.any():
        allowed &= ~cv2.dilate(sclera.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        search &= allowed
    if sclera is not None and sclera.any() and geometry:
        ys, xs = np.where(sclera)
        search &= allowed & (yy > np.percentile(ys, 40))
        if iris is None:
            x0, x1 = np.percentile(xs, [2, 98])
            pad = 0.25 * (x1 - x0)
            search &= (xx > x0 - pad) & (xx < x1 + pad) & (yy < ys.max() + 0.9 * (x1 - x0))
    top = _upper_edge(h, w, iris, sclera) if geometry else None
    if top is not None:
        # веко оттянуто вниз: слизистая — полоса сразу под склерой/радужкой, толщиной не больше ~1,8 радиуса
        thick = 1.8 * (iris[2] if iris is not None else 0.12 * w)
        search &= (yy > top[None, :] - 0.15 * thick) & (yy < top[None, :] + thick)
    if search.sum() < 50:
        return np.zeros((h, w), bool)
    free = top is None and (iris is None or not geometry)
    t_lo = max(_otsu(R, search), float(np.percentile(R[allowed], 75 if free else 65)))
    t_hi = max(float(np.percentile(R[search], 92)), t_lo)
    grow = search & (R > t_lo)
    n, lab = cv2.connectedComponents(_smooth(grow, 3).astype(np.uint8), 8)
    seeds = np.unique(lab[search & (R >= t_hi)])
    cand = np.isin(lab, seeds[seeds > 0])
    cand = _components(cand, MIN_CONJ_FRAC * h * w * 0.5, 0.3, 2)
    if not cand.any():
        return np.zeros((h, w), bool)
    # GrabCut только в полосе ±6 px вокруг кандидата: уточняет край века, но не утекает в кожу
    band = cv2.dilate(cand.astype(np.uint8), np.ones((13, 13), np.uint8)).astype(bool)
    core = cv2.erode(cand.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    gc = np.full((h, w), cv2.GC_BGD, np.uint8)
    gc[band & allowed] = cv2.GC_PR_BGD
    gc[cand] = cv2.GC_PR_FGD
    gc[core & (R > t_hi)] = cv2.GC_FGD
    m = cand
    if (gc == cv2.GC_FGD).sum() >= 10 and (gc == cv2.GC_PR_BGD).sum() >= 10:
        img = (np.clip(maps['n'], 0, 1) * 255).astype(np.uint8)
        bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(np.ascontiguousarray(img[..., ::-1]), gc, None, bgd, fgd, 3, cv2.GC_INIT_WITH_MASK)
            m = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
        except cv2.error:
            pass
    m = _components(_smooth(m & allowed, 5), MIN_CONJ_FRAC * h * w * 0.5, 0.3, 2)
    return _fill_holes(m) & allowed


# ------------------------------------------------------------------ проверки правдоподобия
def conj_contrast(maps, conj, exclude=None):
    """Насколько найденная ткань краснее окружающей кожи (единицы a*). 0 — не выделяется."""
    if not conj.any():
        return 0.0
    ring = cv2.dilate(conj.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~conj
    ring &= maps['L'] > 25                                # сравниваем с кожей: без ресниц и теней
    if exclude is not None:
        ring &= ~cv2.dilate(exclude.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)   # и без склеры
    if ring.sum() < 20:
        return 0.0
    R = redness(maps)
    return max(0.0, float(np.median(R[conj]) - np.median(R[ring])))


def check_conj(maps, conj, cutout=False, sclera=None):
    """-> (ok, причина). Конъюнктива должна быть розово-красной и не крошечной. Для уже вырезанной ткани
    оттенок не проверяется: пользователь сам показал ткань, а свет мог сдвинуть цвет."""
    h, w = conj.shape
    if conj.sum() < (0.003 if cutout else MIN_CONJ_FRAC) * h * w:
        return False, 'конъюнктива не видна или слишком мала'
    a, hab = np.median(maps['a_raw'][conj]), np.median(maps['hab'][conj])
    if a < 2:
        return False, f'найденная область не похожа на слизистую (a*={a:.0f}, нет красного)'
    if not cutout and not (-25 < hab < 75):
        return False, f'оттенок не как у слизистой (h={hab:.0f}°)'
    if not cutout and conj_contrast(maps, conj, sclera) < 2:
        return False, 'найденная область не краснее соседней кожи — это не слизистая'
    return True, ''


def check_sclera(maps, sclera, conj=None, cutout=False):
    h, w = sclera.shape
    if sclera.sum() < MIN_SCLERA_FRAC * h * w:
        return False, 'склера не видна или слишком мала'
    C, L = np.median(maps['C_white'][sclera]), np.median(maps['L'][sclera])
    if C > 20 or L < 35:
        return False, f'найденная область не похожа на склеру (C*={C:.0f}, L*={L:.0f})'
    if cutout and np.median(maps['a_raw'][sclera]) > 12:    # у вырезки нет соседних тканей для сравнения
        return False, 'найденная область красная — похоже на конъюнктиву, а не на склеру'
    if conj is not None and conj.any() and np.median(maps['L'][conj]) > L + 5:
        return False, '«склера» темнее конъюнктивы — вероятно, это тень или кожа'
    return True, ''


# ------------------------------------------------------------------ глаз на снимке
def _prepare(rgb, side=WORK_SIDE):
    h, w = rgb.shape[:2]
    s = side / max(h, w)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC
    return cv2.resize(np.ascontiguousarray(rgb, np.float32), (max(1, round(w * s)), max(1, round(h * s))),
                      interpolation=interp)


def face_eye_crops(rgb, landmarker):
    """Вырезки глаз по точкам лица MediaPipe (если лицо в кадре). Кроп с запасом вниз: оттянутое веко
    уходит ниже контура глаза. -> список словарей {how, box, iris (в координатах кадра), eye_poly}."""
    if not landmarker:
        return []
    h, w = rgb.shape[:2]
    img = np.ascontiguousarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    try:
        import mediapipe as mp
        img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    except Exception:
        pass                                   # без mediapipe (тест с подменой детектора) отдаём массив
    try:
        res = landmarker.detect(img)
    except Exception:
        return []
    if not res.face_landmarks:
        return []
    P = np.array([[p.x * w, p.y * h] for p in res.face_landmarks[0]])
    out = []
    for side, eye, iris in (('правый', R_EYE, R_IRIS), ('левый', L_EYE, L_IRIS)):
        e = P[eye]
        x0, x1 = e[:, 0].min(), e[:, 0].max()
        W = max(x1 - x0, 10)
        cx, cy = (x0 + x1) / 2, e[:, 1].mean()
        box = [int(max(0, cx - 0.9 * W)), int(max(0, cy - 0.6 * W)), int(min(w, cx + 0.9 * W)),
               int(min(h, cy + 1.0 * W))]
        ir = None
        if len(P) > max(iris):
            c = P[iris[0]]
            ir = (float(c[0]), float(c[1]), float(np.mean(np.hypot(*(P[iris[1:]] - c).T))))
        if box[2] - box[0] > 20 and box[3] - box[1] > 20:
            out.append({'how': f'лицо, {side} глаз', 'box': box, 'iris': ir, 'eye_poly': e})
    return out


def detect_regions(rgb, mode='auto', landmarker=None):
    """Главная функция: снимок -> вырезка, маски тканей и объяснение.

    Возвращает dict:
        crop      — вырезка (float32 RGB, длинная сторона WORK_SIDE), на ней же маски;
        conj, sclera — bool-маски (пустые, если ткань не найдена или не прошла проверку);
        kind      — что на снимке: 'вырезанная ткань' / 'крупный план века' / 'глаз' / 'лицо, ... глаз';
        use       — какие ткани пригодны: подмножество {'conj', 'sclera'};
        notes     — список пояснений для пользователя (почему ткань не используется);
        src_side  — размер вырезки на исходном фото, px; iris — (cx, cy, r) в координатах вырезки или None.
    """
    if mode not in MODES:
        raise ValueError(f'mode должен быть одним из {MODES}')
    want_c, want_s = mode in ('auto', 'conj', 'both'), mode in ('auto', 'sclera', 'both')
    H, W = rgb.shape[:2]
    best = None
    faces = face_eye_crops(rgb, landmarker)
    for c in faces:
        x0, y0, x1, y1 = c['box']
        crop = _prepare(rgb[y0:y1, x0:x1])
        s = crop.shape[1] / (x1 - x0)
        iris = None if c['iris'] is None else ((c['iris'][0] - x0) * s, (c['iris'][1] - y0) * s, c['iris'][2] * s)
        r = _analyze_crop(crop, want_c, want_s, iris=iris, kind=c['how'])
        r['src_side'] = max(x1 - x0, y1 - y0)
        if best is None or r['score'] > best['score']:
            best = r
    closeup = faces and max(c['box'][2] - c['box'][0] for c in faces) > 0.5 * W
    if not faces or (closeup and not best['use']):
        # лица нет — это крупный план глаза/века или вырезанная ткань. Если в кадре всё лицо, весь кадр
        # не разбираем: иначе за «конъюнктиву» можно принять губы, а за «склеру» — фон
        r = _analyze_crop(_prepare(rgb), want_c, want_s)
        r['src_side'] = max(H, W)
        if best is None or r['score'] > best['score']:
            best = r
    if mode == 'auto' and best['use']:
        best['notes'] = []                       # в «Авто» отсутствие второй ткани — не проблема
    if mode == 'both' and len(best['use']) == 1:
        best['notes'].append('оценка по одной ткани')
    return best


def _analyze_crop(crop, want_c, want_s, iris=None, kind=None):
    maps = color_maps(crop)
    h, w = crop.shape[:2]
    notes = []
    bg = cutout_background(crop)
    conj = np.zeros((h, w), bool)
    sclera = np.zeros((h, w), bool)
    if bg is not None:
        # ткань уже вырезана (как в CP-AnemiC): всё, что не фон, — одна ткань; какая — по цвету
        tissue = _components(_smooth(~bg, 3), 0.01 * h * w, 0.2, 3)
        is_white = np.median(maps['C_raw'][tissue]) < 9 and np.median(maps['L_raw'][tissue]) > 55
        if want_s and (is_white or not want_c):
            sclera = tissue.copy()
        elif want_c:
            conj = tissue.copy()
        kind = 'вырезанная ткань'
    else:
        if iris is None:
            iris = find_iris(crop, maps)
        if iris is None:
            kind = kind or 'крупный план (радужка не видна)'
        else:
            kind = kind or 'глаз крупным планом'
        # склеру ищем всегда: даже в режиме «только конъюнктива» она подсказывает, где край века
        sclera = find_sclera(crop, maps, iris)
        if want_c:
            conj = find_conj(crop, maps, iris, sclera if sclera.any() else None)
            if iris is not None or sclera.any():
                # радужка/склера могли найтись ошибочно (крупный план века) — ищем слизистую и по всему кадру
                # и берём вариант, где ткань заметнее краснее своего окружения
                alt = find_conj(crop, maps, iris, sclera if sclera.any() else None, geometry=False)
                s_geo = conj_contrast(maps, conj, sclera) * 1.5 if check_conj(maps, conj, sclera=sclera)[0] else -1
                s_alt = conj_contrast(maps, alt, sclera) if check_conj(maps, alt, sclera=sclera)[0] else -1
                if s_alt > s_geo:
                    conj = alt
            if conj.any() and sclera.any():
                sclera &= ~conj
    sclera_found = sclera.copy()
    if not want_s:
        sclera[:] = False
    use = set()
    if want_c:
        ok, why = check_conj(maps, conj, cutout=bg is not None, sclera=sclera_found)
        if ok:
            use.add('conj')
        else:
            conj[:] = False
            notes.append(why)
    if want_s:
        ok, why = check_sclera(maps, sclera, conj, cutout=bg is not None)
        if ok:
            use.add('sclera')
        else:
            sclera[:] = False
            notes.append(why)
    score = (2.0 if 'conj' in use else 0) + (1.0 if 'sclera' in use else 0) + 0.1 * (iris is not None)
    return {'crop': crop, 'conj': conj, 'sclera': sclera, 'kind': kind, 'use': use, 'notes': notes,
            'iris': iris, 'score': score}


# ------------------------------------------------------------------ контроль качества
def blur_ratio(crop, roi):
    """Резкость без привязки к масштабу и текстуре: во сколько раз контуры (|лапласиан|) у границ тканей
    сильнее, чем у той же картинки после размытия σ=1,5. Резкий снимок ≥ ~1,8, размытый ≈ 1,1.
    (В v11 мерили дисперсию лапласиана по всей ткани — гладкая слизистая выглядела «размытой».)"""
    g = cv2.cvtColor(np.clip(crop, 0, 1).astype(np.float32), cv2.COLOR_RGB2GRAY)
    band = cv2.dilate(roi.astype(np.uint8), np.ones((9, 9), np.uint8)) > 0
    e0 = np.abs(cv2.Laplacian(g, cv2.CV_32F))[band].mean()
    e1 = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.5), cv2.CV_32F))[band].mean()
    return float(e0 / (e1 + 1e-6))


def quality(crop, conj, sclera, src_side):
    """-> список проблем, при которых оценка ненадёжна."""
    issues = []
    if src_side < 64:
        issues.append('ткань слишком мелкая в кадре — снимите ближе')
    roi = conj | sclera
    if not roi.any():
        issues.append('не видно ни конъюнктивы, ни склеры')
        return issues
    if blur_ratio(crop, roi) < 1.3:
        issues.append('снимок размыт')
    v = crop.max(-1)[roi]
    if np.median(v) < 0.12:
        issues.append('слишком темно')
    if (crop.min(-1)[roi] > 0.98).mean() > 0.3:       # все три канала в насыщении — цвет ткани потерян
        issues.append('пересвет / вспышка')
    return issues
