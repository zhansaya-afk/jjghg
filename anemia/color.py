"""Цветовые пространства и индексы бледности.

Соглашение: изображения — float32 RGB в [0, 1] (sRGB), маски — bool той же формы H×W.
Формулы перенесены из прототипов v10/v11 (md/current/), каждая покрыта тестом в tests/test_color.py.
"""
import cv2
import numpy as np

EPS = 1e-4


def srgb_to_lin(x):
    x = np.asarray(x, np.float32)
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4).astype(np.float32)


def lin_to_srgb(x):
    x = np.clip(np.asarray(x, np.float32), 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * x ** (1 / 2.4) - 0.055).astype(np.float32)


def rgb_to_lab(rgb):
    """CIELAB (D65): L 0..100, a/b примерно −128..127. Считается попиксельно (не Lab от среднего RGB)."""
    return cv2.cvtColor(np.ascontiguousarray(rgb, np.float32), cv2.COLOR_RGB2Lab)


def rgb_to_hsi(rgb):
    """HSI по Гонсалесу: H в градусах 0..360, S 0..1, I 0..1. H почти не зависит от яркости света."""
    rgb = np.asarray(rgb, np.float32)
    R, G, B = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    I = (R + G + B) / 3
    S = 1 - np.minimum(np.minimum(R, G), B) / (I + EPS)
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B))
    # без «+EPS» в знаменателе: в v10 он сдвигал оттенок красных пикселей на ~1° (arccos крут около 1)
    th = np.degrees(np.arccos(np.clip(num / np.maximum(den, 1e-12), -1, 1)))
    th = np.where(den < 1e-8, 0.0, th)          # серый: оттенок не определён, берём 0
    H = np.where(B <= G, th, 360 - th)
    return H.astype(np.float32), np.clip(S, 0, 1).astype(np.float32), I.astype(np.float32)


def erythema_index(rgb):
    """EI = log10(R) − log10(G). Hb поглощает зелёный и почти не поглощает красный → бледность снижает EI."""
    px = np.clip(np.asarray(rgb, np.float32), 1 / 255, 1)
    return (np.log10(px[..., 0]) - np.log10(px[..., 1])).astype(np.float32)


def signed_hue(H):
    """Оттенок в −180..180°, чтобы красный (около 0/360) не рвался на два края."""
    return np.where(H > 180, H - 360, H)


def high_hue_ratio(rgb, mask, half_width=10.0):
    """HHR: доля пикселей ткани с «красным» оттенком |H| < half_width° (HSI)."""
    H, _, _ = rgb_to_hsi(rgb)
    h = signed_hue(H[mask])
    return float(np.mean(np.abs(h) < half_width)) if h.size else float('nan')


def tissue_mask(rgb, mask, min_px=30):
    """Пиксели ткани внутри маски без бликов (светлые и бесцветные), теней и ресниц (тёмные)."""
    v = rgb.max(-1)
    s = (v - rgb.min(-1)) / (v + EPS)
    m = mask & (v > 0.08) & ~((v > 0.92) & (s < 0.25))
    if m.sum() < min_px:
        return mask & (v > 0.03)
    lo, hi = np.percentile(v[m], [3, 99])
    return m & (v >= lo) & (v <= hi)


def white_balance_by(rgb, ref_mask):
    """Баланс белого по эталонной ткани (склера): линейный RGB делится на её медиану (модель фон Криса).
    Любое умножение каналов источником света в линейном пространстве после этого сокращается."""
    lin = srgb_to_lin(np.clip(rgb, 0, 1))
    ref = np.median(lin[ref_mask], axis=0) + 1e-6
    return lin_to_srgb(lin * (ref.mean() / ref))


def roi_features(rgb, mask, prefix=''):
    """Признаки цвета области: EI, хроматические координаты, Lab, оттенок HSI, HHR, контраст сосудов.
    Для каждого канала — среднее, p10/p50/p90 и размах. None — мало чистой ткани."""
    m = tissue_mask(rgb, mask)
    if m.sum() < 30:
        return None
    px = np.clip(rgb[m].astype(np.float32), 1 / 255, 1)
    R, G, B = px[:, 0], px[:, 1], px[:, 2]
    lab = rgb_to_lab(px[None])[0]
    L, A, Bb = lab[:, 0], lab[:, 1], lab[:, 2]
    H, S, _ = rgb_to_hsi(px[None])
    hue = signed_hue(H[0])
    tot = R + G + B
    ch = {'EI': np.log10(R) - np.log10(G),
          'lRB': np.log(R / B), 'lGB': np.log(G / B),
          'r': R / tot, 'g': G / tot, 'a': A, 'b': Bb, 'L': L, 'hue': hue, 'sat': S[0],
          'C': np.hypot(A, Bb), 'hab': np.degrees(np.arctan2(Bb, A))}
    f = {}
    for k, v in ch.items():
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        f.update({k + '_m': float(v.mean()), k + '_p10': float(p10), k + '_p50': float(p50),
                  k + '_p90': float(p90), k + '_iqr': float(p90 - p10)})
    f['HHR'] = float(np.mean(np.abs(hue) < 10))
    f['vessel'] = float(np.percentile(ch['EI'], 95) - np.median(ch['EI']))
    f['n_px'] = int(m.sum())
    return {prefix + k: v for k, v in f.items()}


def relative_features(rgb, conj, sclera):
    """Цвет конъюнктивы относительно склеры того же снимка: свет и камера меняют обе ткани одинаково.
    Отношения считаются в линейном RGB, поэтому множитель источника света сокращается точно.
    None — одной из тканей слишком мало."""
    if conj is None or sclera is None:
        return None
    mc, ms = tissue_mask(rgb, conj), tissue_mask(rgb, sclera)
    if mc.sum() < 30 or ms.sum() < 30:
        return None
    lin = srgb_to_lin(np.clip(rgb, 1 / 255, 1))
    c, s = np.median(lin[mc], axis=0) + 1e-6, np.median(lin[ms], axis=0) + 1e-6
    q = c / s
    return {'rel_lR': float(np.log(q[0])), 'rel_lG': float(np.log(q[1])), 'rel_lB': float(np.log(q[2])),
            'rel_lRG': float(np.log(q[0] / q[1])), 'rel_lRB': float(np.log(q[0] / q[2]))}
