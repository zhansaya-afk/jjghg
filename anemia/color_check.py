"""Сверка формул цвета (anemia/color.py) с эталоном и стабильность признаков при другом свете (2026-11-18).

Эталон — независимая реализация в float64 прямо по стандартам, без OpenCV:
    sRGB → линейный: IEC 61966-2-1; линейный → XYZ: матрица sRGB (D65); XYZ → CIELAB: CIE 15:2004, белая точка
    D65 (Xn = 0,95047, Zn = 1,08883); HSI — Гонсалес и Вудс; EI = log10 R − log10 G.
Опорные точки, известные из литературы (Lindbloom, CIE): красный sRGB → Lab (53,24; 80,09; 67,20), зелёный →
(87,73; −86,18; 83,18), синий → (32,30; 79,19; −107,86), белый → (100; 0; 0); оттенок HSI основных цветов
0/60/120/180/240/300°.
Стабильность — модель света augment.apply_illuminant (фон Крис, баланс белого камеры на D65): это проверка кода и
формул, не точность на реальных снимках. Все функции чистые.
"""
import numpy as np

from . import augment as AU
from . import color as C

SRGB_TO_XYZ = np.array([[0.4124564, 0.3575761, 0.1804375],
                        [0.2126729, 0.7151522, 0.0721750],
                        [0.0193339, 0.1191920, 0.9503041]])
WHITE_D65 = np.array([0.95047, 1.0, 1.08883])

# Опорные значения из литературы (Lindbloom, sRGB D65): sRGB 0–1 → (L, a, b)
LAB_ANCHORS = {'красный': ((1, 0, 0), (53.2408, 80.0925, 67.2032)),
               'зелёный': ((0, 1, 0), (87.7347, -86.1827, 83.1793)),
               'синий': ((0, 0, 1), (32.2970, 79.1875, -107.8602)),
               'белый': ((1, 1, 1), (100.0, 0.0, 0.0)),
               'чёрный': ((0, 0, 0), (0.0, 0.0, 0.0))}
HUE_ANCHORS = {'красный': ((1, 0, 0), 0.0), 'жёлтый': ((1, 1, 0), 60.0), 'зелёный': ((0, 1, 0), 120.0),
               'голубой': ((0, 1, 1), 180.0), 'синий': ((0, 0, 1), 240.0), 'пурпурный': ((1, 0, 1), 300.0)}

# ColorChecker Classic, приближённые 8-бит sRGB (BabelColor). Lab для них считается эталоном ниже, а не берётся из
# таблиц: опубликованные Lab — при D50, у нас D65.
COLORCHECKER = {
    'dark skin': (115, 82, 68), 'light skin': (194, 150, 130), 'blue sky': (98, 122, 157), 'foliage': (87, 108, 67),
    'blue flower': (133, 128, 177), 'bluish green': (103, 189, 170), 'orange': (214, 126, 44),
    'purplish blue': (80, 91, 166), 'moderate red': (193, 90, 99), 'purple': (94, 60, 108),
    'yellow green': (157, 188, 64), 'orange yellow': (224, 163, 46), 'blue': (56, 61, 150), 'green': (70, 148, 73),
    'red': (175, 54, 60), 'yellow': (231, 199, 31), 'magenta': (187, 86, 149), 'cyan': (8, 133, 161),
    'white 9.5': (243, 243, 242), 'neutral 8': (200, 200, 200), 'neutral 6.5': (160, 160, 160),
    'neutral 5': (122, 122, 121), 'neutral 3.5': (85, 85, 85), 'black 2': (52, 52, 52)}

# Цвета тканей для графика стабильности (sRGB 0–1, типичные значения со снимков-насадок; не пациенты)
TISSUES = {'конъюнктива, красная': (0.78, 0.36, 0.36), 'конъюнктива, бледная': (0.86, 0.62, 0.60),
           'склера': (0.85, 0.84, 0.86)}


def ref_lin(x):
    x = np.asarray(x, np.float64)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def ref_lab(rgb):
    """sRGB 0–1 (…×3) → CIELAB D65, float64, по CIE 15:2004."""
    xyz = ref_lin(rgb) @ SRGB_TO_XYZ.T / WHITE_D65
    d = 6 / 29
    f = np.where(xyz > d ** 3, np.cbrt(xyz), xyz / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)


def ref_hsi(rgb):
    """Гонсалес и Вудс, float64: H° 0–360, S, I."""
    rgb = np.asarray(rgb, np.float64)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    I = (R + G + B) / 3
    S = np.where(I > 0, 1 - rgb.min(-1) / np.where(I > 0, I, 1), 0)
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    th = np.where(den > 1e-12, np.degrees(np.arccos(np.clip(num / np.where(den > 1e-12, den, 1), -1, 1))), 0)
    return np.where(B <= G, th, 360 - th), S, I


def ref_ei(rgb):
    px = np.clip(np.asarray(rgb, np.float64), 1 / 255, 1)
    return np.log10(px[..., 0]) - np.log10(px[..., 1])


def delta_e76(a, b):
    return np.sqrt(((np.asarray(a, float) - np.asarray(b, float)) ** 2).sum(-1))


def hue_diff(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 360
    return np.minimum(d, 360 - d)


def verify(n_random=20000, seed=0):
    """→ список строк таблицы: что сверяли, сколько цветов, макс. ошибка, допуск, прошло ли."""
    rows = []

    def add(what, n, err, tol, unit):
        rows.append({'проверка': what, 'n': int(n), 'макс_ошибка': float(err), 'допуск': tol, 'единица': unit,
                     'прошла': bool(err <= tol)})

    a = np.array([v[0] for v in LAB_ANCHORS.values()], np.float32)
    lab_lit = np.array([v[1] for v in LAB_ANCHORS.values()])
    add('Lab кода vs литература (основные цвета, белый, чёрный)', len(a),
        delta_e76(C.rgb_to_lab(a[None])[0], lab_lit).max(), 0.01, 'ΔE76')
    add('Lab эталона float64 vs литература', len(a), delta_e76(ref_lab(a), lab_lit).max(), 0.001, 'ΔE76')
    cc = np.array(list(COLORCHECKER.values()), np.float32) / 255
    # OpenCV (float32) считает Lab со своими константами и приближениями: сдвиг ≈ 0,14 ΔE по всему кубу. Допуск —
    # 1 ΔE (порог различимости глазом ~1–2,3); сдвиг одинаков при обучении и применении, на модель не влияет.
    add('Lab кода vs эталон: 24 поля ColorChecker', len(cc), delta_e76(C.rgb_to_lab(cc[None])[0], ref_lab(cc)).max(),
        1.0, 'ΔE76')
    tis = np.array(list(TISSUES.values()), np.float32)
    add('Lab кода vs эталон: цвета тканей (конъюнктива, склера)', len(tis),
        delta_e76(C.rgb_to_lab(tis[None])[0], ref_lab(tis)).max(), 1.0, 'ΔE76')
    rnd = np.random.default_rng(seed).random((n_random, 3)).astype(np.float32)
    add(f'Lab кода vs эталон: {n_random} случайных цветов', n_random,
        delta_e76(C.rgb_to_lab(rnd[None])[0], ref_lab(rnd)).max(), 1.0, 'ΔE76')
    add(f'то же, медиана', n_random, np.median(delta_e76(C.rgb_to_lab(rnd[None])[0], ref_lab(rnd))), 1.0, 'ΔE76')

    h = np.array([v[0] for v in HUE_ANCHORS.values()], np.float32)
    hl = np.array([v[1] for v in HUE_ANCHORS.values()])
    H, S, _ = C.rgb_to_hsi(h[None])
    add('Оттенок HSI кода vs 0/60/…/300° (основные и дополнительные)', len(h), hue_diff(H[0], hl).max(), 0.01, '°')
    add('Насыщенность HSI основных цветов = 1', 3, np.abs(S[0][[0, 2, 4]] - 1).max(), 1e-4, '')
    chrom = rnd[(rnd.max(1) - rnd.min(1)) > 0.02]            # у почти серых оттенок не определён
    Hc, Sc, Ic = C.rgb_to_hsi(chrom[None])
    Hr, Sr, Ir = ref_hsi(chrom)
    add('Оттенок HSI кода vs эталон (случайные цветные)', len(chrom), hue_diff(Hc[0], Hr).max(), 0.05, '°')
    # S кода = 1 − min/(I + 1e-4): у тёмных пикселей на ~0,001 меньше точного; в признаки такие не попадают (V > 0,08)
    add('S и I HSI кода vs эталон', len(chrom), max(np.abs(Sc[0] - Sr).max(), np.abs(Ic[0] - Ir).max()), 0.002, '')
    blk = C.rgb_to_hsi(np.zeros((1, 1, 3), np.float32))[1][0, 0]
    rows.append({'проверка': 'Чёрный (0,0,0): S HSI кода (по Гонсалесу не определена; tissue_mask такие пиксели '
                             'отбрасывает)', 'n': 1, 'макс_ошибка': float(blk), 'допуск': 'особенность', 'единица': '',
                 'прошла': True})
    g = np.linspace(0.1, 1, 10, dtype=np.float32)
    gray = np.stack([g, g, g], -1)
    add('Серые 0,1–1: S HSI ≈ 0, a* = b* ≈ 0', len(g), max(C.rgb_to_hsi(gray[None])[1].max(),
                                                      np.abs(C.rgb_to_lab(gray[None])[0][:, 1:]).max()), 0.1, '')

    add('EI кода vs эталон log10 R − log10 G', n_random, np.abs(C.erythema_index(rnd) - ref_ei(rnd)).max(), 1e-5, '')
    ei_pts = np.array([[0.5, 0.5, 0.2], [0.5, 0.05, 0.1], [0.1, 1.0, 0.3]], np.float32)
    add('EI опорные точки: R=G → 0; R/G=10 → 1; R/G=0,1 → −1', 3,
        np.abs(C.erythema_index(ei_pts) - np.array([0, 1, -1])).max(), 1e-5, '')

    lin = C.srgb_to_lin(rnd)
    add('sRGB → линейный → sRGB (обратимость)', n_random, np.abs(C.lin_to_srgb(lin) - rnd).max(), 1e-5, '')
    add('sRGB → линейный кода vs IEC 61966-2-1', n_random, np.abs(lin - ref_lin(rnd)).max(), 1e-6, '')

    # HHR: патч с известной долей «красных» пикселей (|H| < 10°)
    k = 37
    patch = np.zeros((10, 10, 3), np.float32)
    patch[...] = (0.3, 0.6, 0.3)                              # зелёный, H = 120°
    patch.reshape(-1, 3)[:k] = (0.8, 0.35, 0.33)               # красный, |H| ≈ 2°
    add(f'HHR: патч с {k}% красных пикселей', 100,
        abs(C.high_hue_ratio(patch, np.ones((10, 10), bool)) - k / 100), 1e-9, 'доля')
    return rows


def tissue_feats(rgb):
    """Цвет одного пикселя ткани (sRGB 0–1) → EI, a*, оттенок HSI (со знаком)."""
    px = np.asarray(rgb, np.float32).reshape(1, 1, 3)
    H, _, _ = C.rgb_to_hsi(px)
    return {'EI': float(C.erythema_index(px)[0, 0]), 'a*': float(C.rgb_to_lab(px)[0, 0, 1]),
            'оттенок': float(C.signed_hue(H)[0, 0])}


def relight(rgb, T, exposure=1.0):
    return np.clip(AU.apply_illuminant(np.asarray(rgb, np.float32).reshape(1, 1, 3), T, exposure), 0, 1)[0, 0]


def white_balanced(conj, sclera):
    """Конъюнктива после баланса белого по склере того же снимка (color.white_balance_by на 2 пикселях)."""
    img = np.array([[conj, sclera]], np.float32)
    return C.white_balance_by(img, np.array([[False, True]]))[0, 0]


def stability(temps=range(2700, 7501, 200), exposures=(0.6, 0.8, 1.0, 1.25, 1.5)):
    """→ строки: ткань, условие, T, экспозиция, способ ('сырой' / 'баланс по склере'), признаки."""
    rows = []
    s0 = TISSUES['склера']
    conds = list(dict.fromkeys([(T, 1.0) for T in temps] + [(6500, e) for e in exposures]
                               + [tuple(c) for c in AU.STRESS_CONDITIONS]))
    for name, c0 in TISSUES.items():
        if name == 'склера':
            continue
        for T, e in conds:
            c, s = relight(c0, T, e), relight(s0, T, e)
            for how, px in (('сырой', c), ('баланс по склере', white_balanced(c, s))):
                rows.append({'ткань': name, 'T': T, 'экспозиция': e, 'способ': how, **tissue_feats(px)})
    return rows


def spread(rows):
    """Размах (max − min) каждого признака по 8 условиям STRESS_CONDITIONS; плюс разница красная − бледная при
    6500 K — чтобы видеть, больше ли разброс от света, чем сам сигнал бледности."""
    import pandas as pd
    T = pd.DataFrame(rows)
    keys = set(AU.STRESS_CONDITIONS)
    T = T[[(t, e) in keys for t, e in zip(T['T'], T['экспозиция'])]]
    out = []
    feats = ['EI', 'a*', 'оттенок']
    ref = T[(T['T'] == 6500) & (T['экспозиция'] == 1.0)]
    for how, g in T.groupby('способ', sort=False):
        r = ref[ref['способ'] == how].set_index('ткань')
        signal = (r.loc['конъюнктива, красная', feats] - r.loc['конъюнктива, бледная', feats]).abs()
        for name, gg in g.groupby('ткань', sort=False):
            d = {'способ': how, 'ткань': name}
            for f in feats:
                d[f + '_размах'] = float(gg[f].max() - gg[f].min())
                d[f + '_сигнал'] = float(signal[f])
            out.append(d)
    return out
