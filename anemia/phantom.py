"""Фантом-эксперимент: печатная цветовая карточка + шкала образцов «бледно-розовый → тёмно-красный» (2026-11-18).

Что проверяет: камеру телефона и коррекцию цвета по карточке — сохраняется ли порядок образцов по a* и
Erythema Index и насколько разбегаются цифры одного образца при другом свете и другом телефоне.
Чего НЕ проверяет: связь цвета с Hb (на бумаге нет крови), сегментацию века, модель риска. Модель не меняется.

Лист A4 (альбом): 4 маркера ArUco DICT_4X4_50 (id 0–3) по углам → гомография мм → пиксели снимка; ряд эталонных
полей (белый, 4 серых, чёрный, 6 цветных), ряд печатных образцов S1–S8, ряд кругов G1–G8 под желатин (по желанию).
Эталон коррекции — номинальные sRGB полей. Принтер печатает их не точно, но одинаково на всех снимках, поэтому
разброс между снимками измеряется честно; абсолютные a*/EI зависят от принтера и бумаги.

Коррекции: raw — как снял телефон; wb — баланс белого по серым полям (3 коэффициента в линейном RGB);
ccm — аффинная матрица 3×4 в линейном RGB по всем непересвеченным эталонным полям (МНК).
Все функции чистые: вход — изображение/таблица, выход — числа.
"""
import re

import cv2
import numpy as np
import pandas as pd

from . import color as C

VERSION = 'phantom-card-v1'
PAGE_MM = (297.0, 210.0)                 # A4 альбом, ширина × высота
MARKER_MM = 22.0
MARGIN_MM = 12.0
ARUCO_DICT = 'DICT_4X4_50'

# Эталонные поля: имя → номинальный sRGB 8 бит. Серые/цветные — значения ColorChecker (BabelColor), белый = бумага.
REF_PATCHES = [('white', (255, 255, 255)), ('n8', (200, 200, 200)), ('n6.5', (160, 160, 160)),
               ('n5', (122, 122, 121)), ('n3.5', (85, 85, 85)), ('black', (35, 35, 35)),
               ('red', (175, 54, 60)), ('green', (70, 148, 73)), ('blue', (56, 61, 150)),
               ('yellow', (231, 199, 31)), ('light_skin', (194, 150, 130)), ('moderate_red', (193, 90, 99))]
NEUTRALS = ('white', 'n8', 'n6.5', 'n5', 'n3.5')
REF_Y, REF_SIZE, REF_GAP = 42.0, 18.0, 3.0
SAMPLE_Y, SAMPLE_SIZE, SAMPLE_GAP = 72.0, 28.0, 5.0
GEL_Y, GEL_D = 118.0, 26.0
N_SAMPLES = 8
SAMPLE_PALE = (0.95, 0.82, 0.80)         # sRGB 0–1, светлее самой бледной конъюнктивы
SAMPLE_DARK = (0.62, 0.16, 0.20)         # темнее самой красной
CLIP_HI, CLIP_LO = 0.98, 0.01            # доля канала: выше/ниже — поле пересвечено/провалено
MIN_FIT_PATCHES = 8
MAX_CLIPPED = 0.2                        # образец с > 20% пересвеченных пикселей исключается из сводки
NAME_RE = re.compile(r'^(?P<phone>[^_\s]+)_(?P<light>[^_\s]+)_(?P<rep>\d+)', re.I)


# ---------- макет листа ----------

def sample_colors(n=N_SAMPLES, pale=SAMPLE_PALE, dark=SAMPLE_DARK):
    """n цветов шкалы: равные шаги в CIELAB от бледного к тёмно-красному → sRGB 0–1 (n×3). a* и EI растут."""
    lp = C.rgb_to_lab(np.array(pale, np.float32)[None, None])[0, 0]
    ld = C.rgb_to_lab(np.array(dark, np.float32)[None, None])[0, 0]
    lab = np.array([(1 - t) * lp + t * ld for t in np.linspace(0, 1, n)], np.float32)[None]
    return np.clip(cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)[0], 0, 1)


def _row_x(n, size, gap):
    w = n * size + (n - 1) * gap
    x0 = (PAGE_MM[0] - w) / 2
    return [x0 + i * (size + gap) for i in range(n)]


def layout():
    """Все элементы листа в мм: список словарей {kind, name, x, y, w, h, rgb(0–1) или None, shape}."""
    items = []
    for x, (name, rgb) in zip(_row_x(len(REF_PATCHES), REF_SIZE, REF_GAP), REF_PATCHES):
        items.append(dict(kind='ref', name=name, x=x, y=REF_Y, w=REF_SIZE, h=REF_SIZE,
                          rgb=np.array(rgb, np.float32) / 255, shape='rect'))
    for i, (x, rgb) in enumerate(zip(_row_x(N_SAMPLES, SAMPLE_SIZE, SAMPLE_GAP), sample_colors())):
        items.append(dict(kind='sample', name=f'S{i + 1}', x=x, y=SAMPLE_Y, w=SAMPLE_SIZE, h=SAMPLE_SIZE,
                          rgb=rgb, shape='rect'))
    for i, x in enumerate(_row_x(N_SAMPLES, SAMPLE_SIZE, SAMPLE_GAP)):
        c = x + SAMPLE_SIZE / 2
        items.append(dict(kind='gel', name=f'G{i + 1}', x=c - GEL_D / 2, y=GEL_Y, w=GEL_D, h=GEL_D,
                          rgb=None, shape='circle'))
    return items


def marker_corners_mm():
    """id → 4 угла маркера в мм (порядок OpenCV: TL, TR, BR, BL)."""
    W, H = PAGE_MM
    m, s = MARGIN_MM, MARKER_MM
    tl = {0: (m, m), 1: (W - m - s, m), 2: (W - m - s, H - m - s), 3: (m, H - m - s)}
    return {i: np.array([(x, y), (x + s, y), (x + s, y + s), (x, y + s)], np.float32) for i, (x, y) in tl.items()}


def _dictionary():
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, ARUCO_DICT))


def _marker_image(i, px):
    d = _dictionary()
    if hasattr(cv2.aruco, 'generateImageMarker'):
        return cv2.aruco.generateImageMarker(d, i, px)
    return cv2.aruco.drawMarker(d, i, px)          # OpenCV < 4.7


def render_sheet(dpi=300):
    """Лист как uint8 RGB (H×W×3) в масштабе 1:1 при данном dpi. Подписи — латиница (шрифт OpenCV)."""
    k = dpi / 25.4
    W, H = int(round(PAGE_MM[0] * k)), int(round(PAGE_MM[1] * k))
    img = np.full((H, W, 3), 255, np.uint8)
    P = lambda v: int(round(v * k))                                    # noqa: E731
    for i, c in marker_corners_mm().items():
        x, y = P(c[0, 0]), P(c[0, 1])
        s = P(MARKER_MM)
        img[y:y + s, x:x + s] = cv2.cvtColor(_marker_image(i, s), cv2.COLOR_GRAY2RGB)
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.0105 * dpi / 3, max(1, dpi // 150)
    for it in layout():
        x0, y0, x1, y1 = P(it['x']), P(it['y']), P(it['x'] + it['w']), P(it['y'] + it['h'])
        if it['shape'] == 'rect':
            img[y0:y1, x0:x1] = np.round(it['rgb'] * 255).astype(np.uint8)
            if it['name'] == 'white':                                   # белое поле видно на бумаге
                cv2.rectangle(img, (x0, y0), (x1, y1), (170, 170, 170), th)
        else:
            cv2.circle(img, ((x0 + x1) // 2, (y0 + y1) // 2), (x1 - x0) // 2, (170, 170, 170), th, cv2.LINE_AA)
        cv2.putText(img, it['name'], (x0, y1 + P(4.5)), font, fs * (0.8 if it['kind'] == 'ref' else 1.0),
                    (60, 60, 60), th, cv2.LINE_AA)
    cv2.putText(img, f'HemaVision {VERSION}  |  print at 100% (no "fit to page"), matte paper',
                (P(40), P(22)), font, fs * 1.1, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(img, 'S1..S8 = printed scale, pale -> dark red.  G1..G8 = place gelatin samples (optional)',
                (P(40), P(30)), font, fs * 0.9, (60, 60, 60), th, cv2.LINE_AA)
    # линейка 50 мм: после печати проверить линейкой (должно быть ровно 50 мм)
    y = P(162)
    cv2.line(img, (P(40), y), (P(90), y), (0, 0, 0), th * 2)
    for t in range(0, 51, 10):
        cv2.line(img, (P(40 + t), y - P(2)), (P(40 + t), y + P(2)), (0, 0, 0), th)
    cv2.putText(img, '50 mm', (P(95), y + P(1.5)), font, fs, (0, 0, 0), th, cv2.LINE_AA)
    cv2.putText(img, 'phone_light_NN.jpg  e.g. p1_okno_01.jpg', (P(130), y + P(1.5)), font, fs, (60, 60, 60), th,
                cv2.LINE_AA)
    return img


def save_sheet_pdf(path, dpi=300):
    """PDF A4 в натуральную величину (Pillow сохраняет разрешение → размер страницы 297×210 мм)."""
    from PIL import Image
    Image.fromarray(render_sheet(dpi)).save(path, 'PDF', resolution=float(dpi))
    return path


# ---------- поиск листа на снимке ----------

def detect_markers(rgb_u8, max_side=2000):
    """id → 4 угла в пикселях исходного снимка. Ищет на уменьшенной копии (быстро на 12 Мп)."""
    g = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    s = min(1.0, max_side / max(g.shape))
    small = cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else g
    d = _dictionary()
    if hasattr(cv2.aruco, 'ArucoDetector'):
        corners, ids, _ = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters()).detectMarkers(small)
    else:                                                               # OpenCV < 4.7
        corners, ids, _ = cv2.aruco.detectMarkers(small, d)
    out = {}
    if ids is not None:
        for c, i in zip(corners, ids.ravel()):
            if 0 <= int(i) <= 3 and int(i) not in out:
                out[int(i)] = c.reshape(4, 2) / s
    return out


def sheet_homography(markers):
    """Гомография мм → пиксели по углам найденных маркеров (нужно ≥ 3 из 4)."""
    if len(markers) < 3:
        raise ValueError(f'на снимке найдено {len(markers)} из 4 маркеров (нужно ≥ 3): весь лист в кадре, без бликов')
    ref = marker_corners_mm()
    src = np.concatenate([ref[i] for i in sorted(markers)])
    dst = np.concatenate([markers[i] for i in sorted(markers)]).astype(np.float32)
    Hm, _ = cv2.findHomography(src, dst, 0)
    return Hm


def _grid_mm(it, frac=0.6, n=25):
    cx, cy = it['x'] + it['w'] / 2, it['y'] + it['h'] / 2
    u = np.linspace(-0.5, 0.5, n) * frac
    gx, gy = np.meshgrid(cx + u * it['w'], cy + u * it['h'])
    pts = np.stack([gx.ravel(), gy.ravel()], -1)
    if it['shape'] == 'circle':                                         # центральная часть круга
        pts = pts[((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2) <= (frac * it['w'] / 2) ** 2]
    return pts.astype(np.float32)


def sample_patches(rgb01, Hm, items=None, frac=0.6):
    """Медиана sRGB (0–1) центральной части каждого поля + доля пересвеченных пикселей. → DataFrame."""
    items = layout() if items is None else items
    h, w = rgb01.shape[:2]
    rows = []
    for it in items:
        px = cv2.perspectiveTransform(_grid_mm(it, frac)[None], Hm)[0]
        xi, yi = np.round(px[:, 0]).astype(int), np.round(px[:, 1]).astype(int)
        ok = (xi >= 0) & (xi < w) & (yi >= 0) & (yi < h)
        if ok.mean() < 0.9:
            rows.append(dict(kind=it['kind'], name=it['name'], R=np.nan, G=np.nan, B=np.nan, clipped=np.nan))
            continue
        v = rgb01[yi[ok], xi[ok]].reshape(-1, 3)
        med = np.median(v, 0)
        rows.append(dict(kind=it['kind'], name=it['name'], R=med[0], G=med[1], B=med[2],
                         clipped=float(np.mean((v >= CLIP_HI).any(1) | (v <= CLIP_LO).any(1)))))
    return pd.DataFrame(rows)


# ---------- коррекция цвета ----------

def _ref_arrays(patches):
    nominal = {n: np.array(v, np.float64) / 255 for n, v in REF_PATCHES}
    ref = patches[patches.kind == 'ref'].set_index('name')
    meas = ref[['R', 'G', 'B']].to_numpy(np.float64)
    usable = ~np.isnan(meas).any(1) & (meas.max(1) < CLIP_HI) & (meas.min(1) > CLIP_LO) & (ref['clipped'] < 0.2)
    target = np.stack([nominal[n] for n in ref.index])
    return ref.index.to_numpy(), meas, target, usable


def fit_correction(patches, method):
    """Функция sRGB(n×3) → sRGB(n×3) для method ∈ {raw, wb, ccm} и число полей, по которым она подобрана."""
    if method == 'raw':
        return (lambda x: np.asarray(x, np.float64)), 0
    names, meas, target, usable = _ref_arrays(patches)
    M, T = C.srgb_to_lin(meas).astype(np.float64), C.srgb_to_lin(target).astype(np.float64)
    if method == 'wb':
        sel = usable & np.isin(names, NEUTRALS)
        if sel.sum() < 2:
            raise ValueError('мало серых полей без пересвета для баланса белого')
        g = (M[sel] * T[sel]).sum(0) / np.maximum((M[sel] ** 2).sum(0), 1e-12)
        return (lambda x: C.lin_to_srgb(C.srgb_to_lin(x) * g).astype(np.float64)), int(sel.sum())
    if method == 'ccm':
        if usable.sum() < MIN_FIT_PATCHES:
            raise ValueError(f'только {int(usable.sum())} эталонных полей без пересвета (нужно ≥ {MIN_FIT_PATCHES})')
        X = np.c_[M[usable], np.ones(usable.sum())]
        A, *_ = np.linalg.lstsq(X, T[usable], rcond=None)               # 4×3, аффинная
        return (lambda x: C.lin_to_srgb(np.c_[C.srgb_to_lin(x), np.ones(len(x))] @ A).astype(np.float64)), \
            int(usable.sum())
    raise ValueError(method)


def _ccm_out_of_gamut(patches, lin_aug):
    """Образец после ccm вышел за 0–1 в линейном RGB (будет обрезан → шаг сжимается)."""
    names, meas, target, usable = _ref_arrays(patches)
    M, T = C.srgb_to_lin(meas).astype(np.float64), C.srgb_to_lin(target).astype(np.float64)
    A, *_ = np.linalg.lstsq(np.c_[M[usable], np.ones(usable.sum())], T[usable], rcond=None)
    y = lin_aug @ A
    return ((y < -0.005) | (y > 1.005)).any(1)


def color_metrics(srgb):
    """sRGB 0–1 (n×3) → DataFrame L, a, b, EI (EI = log10 R − log10 G, как anemia.color)."""
    x = np.clip(np.asarray(srgb, np.float32), 0, 1)[None]
    lab = C.rgb_to_lab(x)[0]
    return pd.DataFrame({'L': lab[:, 0], 'a': lab[:, 1], 'b': lab[:, 2], 'EI': C.erythema_index(x)[0]})


def card_residual_de(patches, fn):
    """Медиана ΔE76 эталонных полей после коррекции к номиналу (на тех же полях → оптимистично)."""
    names, meas, target, usable = _ref_arrays(patches)
    if not usable.any():
        return float('nan')
    a = C.rgb_to_lab(np.clip(fn(meas[usable]), 0, 1).astype(np.float32)[None])[0]
    b = C.rgb_to_lab(target[usable].astype(np.float32)[None])[0]
    return float(np.median(np.sqrt(((a - b) ** 2).sum(1))))


METHODS = ('raw', 'wb', 'ccm')


def analyze_image(rgb_u8, kinds=('sample',)):
    """Один снимок → DataFrame (method, kind, name, R, G, B, L, a, b, EI, clipped, card_dE, n_fit)."""
    rgb01 = rgb_u8.astype(np.float32) / 255
    patches = sample_patches(rgb01, sheet_homography(detect_markers(rgb_u8)))
    tgt = patches[patches.kind.isin(kinds)].reset_index(drop=True)
    out = []
    for m in METHODS:
        try:
            fn, n_fit = fit_correction(patches, m)
        except ValueError as e:
            out.append(pd.DataFrame([dict(method=m, kind=k, name=None, error=str(e)) for k in kinds]))
            continue
        if m == 'ccm':
            lin = np.c_[C.srgb_to_lin(tgt[['R', 'G', 'B']].to_numpy(np.float64)), np.ones(len(tgt))]
        corr = fn(tgt[['R', 'G', 'B']].to_numpy(np.float64))
        df = pd.concat([tgt[['kind', 'name', 'clipped']], pd.DataFrame(corr, columns=['R', 'G', 'B']),
                        color_metrics(corr)], axis=1)
        df.insert(0, 'method', m)
        df['card_dE'] = card_residual_de(patches, fn)
        df['n_fit'] = n_fit
        df['out_of_gamut'] = False
        if m == 'ccm':
            df['out_of_gamut'] = _ccm_out_of_gamut(patches, lin)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def parse_name(fname):
    """«p1_okno_01.jpg» → (phone, light, rep). Не по шаблону → ('?', '?', 0)."""
    m = NAME_RE.match(fname.rsplit('/', 1)[-1])
    return (m['phone'].lower(), m['light'].lower(), int(m['rep'])) if m else ('?', '?', 0)


# ---------- сводка ----------

def spearman(x, y):
    x, y = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    if np.std(x) == 0 or np.std(y) == 0:
        return float('nan')
    return float(np.corrcoef(x, y)[0, 1])


def _order(name):
    return int(re.sub(r'\D', '', str(name)) or 0)


def summarize(df, metrics=('a', 'EI')):
    """df: строки analyze_image + photo, phone, light (kind одного типа). → сводка по method × metric.

    rho_* — Спирмен «номер образца ↔ метрика» на каждом снимке; monotone — доля снимков, где порядок идеален.
    sd_step — медиана SD одного образца по всем снимкам / медианный шаг между соседними образцами
              (< 0,5 → соседние образцы различимы при любом снимке; это главное число).
    cv_pct — медиана SD/|среднее|·100 по образцам (как просили; при a*, EI около 0 раздувается).
    light_sd_step — то же, но SD между средними по свету внутри одного телефона;
    phone_sd_step — SD между средними по телефонам при одном свете.
    """
    df = df[df.name.notna()].copy()
    df['idx'] = df.name.map(_order)
    n_clipped = df.groupby('method').clipped.apply(lambda c: int((c > MAX_CLIPPED).sum()))
    n_gamut = df.groupby('method').out_of_gamut.apply(lambda c: int(c.fillna(False).astype(bool).sum())) \
        if 'out_of_gamut' in df else pd.Series(dtype=int)
    df = df[~(df.clipped > MAX_CLIPPED)]                    # блик/пересвет на образце — не в сводку
    rows = []
    for m, met in [(m, met) for m in METHODS for met in metrics]:
        d = df[df.method == m]
        if d.empty or met not in d:
            continue
        rho = pd.Series({ph: spearman(g.idx, g[met]) for ph, g in d.groupby('photo')})
        mean = d.groupby('idx')[met].mean().sort_index()
        step = float(np.median(np.diff(mean.to_numpy()))) if len(mean) > 1 else float('nan')   # со знаком
        step = step if step > 0 else float('nan')           # шкала в среднем не растёт → разброс не определён
        sd = d.groupby('idx')[met].std(ddof=1)

        def nested(outer, inner, d=d, met=met, step=step):
            g = d.groupby([outer, inner, 'idx'])[met].mean().reset_index()
            s = g.groupby([outer, 'idx'])[met].std(ddof=1)
            return float(np.nanmedian(s.to_numpy()) / step) if s.notna().any() and step > 0 else float('nan')

        rows.append(dict(method=m, metric=met, n_photos=int(d.photo.nunique()),
                         rho_median=float(rho.median()), rho_min=float(rho.min()),
                         monotone=float(np.mean(rho > 0.9999)),
                         step=step, sd_step=float(sd.median() / step) if step > 0 else float('nan'),
                         cv_pct=float(np.median(100 * sd / np.maximum(mean.abs(), 1e-9))),
                         light_sd_step=nested('phone', 'light'), phone_sd_step=nested('light', 'phone'),
                         card_dE=float(d.groupby('photo').card_dE.first().median()),
                         n_clipped_samples=int(n_clipped.get(m, 0)), n_out_of_gamut=int(n_gamut.get(m, 0))))
    return pd.DataFrame(rows)


# Заранее заданные критерии (зафиксированы до съёмки 2026-11-18, по фото не подбираются)
CRITERIA = {'monotone_min': 0.90, 'sd_step_max': 0.50, 'coverage_min': 0.80}


def verdict(summary, n_total=None, method='ccm', metric='a'):
    """n_total — все снимки в папке, включая те, где лист не найден (иначе критерий охвата не работает)."""
    r = summary[(summary.method == method) & (summary.metric == metric)]
    if r.empty:
        return 'НЕ ПРОЙДЕН (нет данных)'
    r = r.iloc[0]
    raw = summary[(summary.method == 'raw') & (summary.metric == metric)]
    n_all = int(n_total) if n_total is not None else (int(raw.n_photos.iloc[0]) if len(raw) else int(r.n_photos))
    better = bool(len(raw)) and r.sd_step < raw.sd_step.iloc[0]
    ok = (r.monotone >= CRITERIA['monotone_min'] and r.sd_step <= CRITERIA['sd_step_max']
          and r.n_photos >= CRITERIA['coverage_min'] * n_all and better)
    return (('ПРОЙДЕН' if ok else 'НЕ ПРОЙДЕН')
            + f' ({method}, {metric}: карточка применена к {int(r.n_photos)} из {n_all} снимков; порядок верен на '
            f'{r.monotone:.0%}, разброс = {r.sd_step:.2f} шага'
            + ('' if better else '; карточка НЕ уменьшила разброс по сравнению с raw') + ')')


# ---------- синтетика: только проверка кода, не результат эксперимента ----------

PHONES_SYNTH = {'p1': np.eye(3), 'p2': np.array([[1.08, -0.06, -0.02], [-0.04, 1.05, -0.01], [0.0, -0.08, 1.08]])}


def synthetic_photo(sheet_u8, T=6500, exposure=1.0, phone='p1', seed=0, out_px=(1600, 2000)):
    """Лист под планковским светом T (баланс камеры на D65), другой «телефон» = матрица 3×3 в линейном RGB,
    перспектива, шум, JPEG-подобное округление. Проверка, что код находит лист и коррекция работает."""
    from . import augment as AU
    rng = np.random.default_rng(seed)
    x = sheet_u8.astype(np.float32) / 255
    lin = C.srgb_to_lin(x) * AU.kelvin_gains(T) * exposure
    lin = np.clip(lin @ np.asarray(PHONES_SYNTH[phone], np.float32).T, 0, 1)
    x = C.lin_to_srgb(lin)
    h, w = x.shape[:2]
    H_out, W_out = out_px
    sc = 0.8 * min(W_out / w, H_out / h)
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    cx, cy = W_out / 2, H_out / 2
    dst = np.float32([[cx - w * sc / 2, cy - h * sc / 2], [cx + w * sc / 2, cy - h * sc / 2],
                      [cx + w * sc / 2, cy + h * sc / 2], [cx - w * sc / 2, cy + h * sc / 2]])
    dst += rng.uniform(-0.04, 0.04, dst.shape).astype(np.float32) * w * sc
    M = cv2.getPerspectiveTransform(src, dst)
    bg = (0.35, 0.33, 0.30)
    y = cv2.warpPerspective(x, M, (W_out, H_out), flags=cv2.INTER_AREA, borderValue=bg)
    y = y + rng.normal(0, 0.01, y.shape).astype(np.float32)
    return np.round(np.clip(y, 0, 1) * 255).astype(np.uint8)
