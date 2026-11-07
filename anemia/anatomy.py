"""Анатомические ограничения масок тканей (2026-10-31).

Конъюнктива нижнего века — одна полоса слизистой между краем века и глазным яблоком: она тянется под глазной щелью
(широкая и низкая), а не островки по кадру. Склера — белок глазного яблока: он лежит над этой полосой и в её
пределах по горизонтали, а не под веком и не сбоку от глаза. Сегментатор века (segment_baseline) и пороговая склера
(regions) на снимках шире обучающих (лицо, пальцы) отмечали кожу и пальцы; эти правила убирают такие части, а если
от маски века остаётся сомнительная часть — анализ отклоняется (лучше отказ, чем признаки цвета кожи).

Единица длины — ширина главной полосы конъюнктивы (не радиус радужки: радужка на телефонных фото бывает ложной).
Пороги (LIMITS) получены fit_limits по маскам врачей и пороговой склере обучающих снимков (Eyes-defy + 83 телефонных
фото с масками), без Hb и меток анемии; scripts/evaluate.py пересчитывает их на обучающих фолдах каждого теста.
Все функции чистые: маски на входе, числа и маски на выходе.
"""
import cv2
import numpy as np

SHARE_MIN = 0.5          # главная полоса — больше половины маски века; иначе неясно, какая часть — веко (принцип)
WFRAC_K = 0.5            # ширина полосы ≥ 0,5 медианы ширины у врачей (доля ширины вырезки)
ASPECT_K = 1.5           # высота/ширина полосы ≤ 1,5 × 98-й перцентиль у врачей: веко — полоса, а не пятно
SCLERA_PAD = 0.15        # запас к перцентилям 2–98% положения склеры (в ширинах полосы)
SCLERA_DROP_MAX = 0.5    # убрано больше половины площади склеры → склера не используется
PART_MIN = 0.1           # части склеры меньше 10% её площади в подбор пределов не идут (шум)

# fit_limits по всем снимкам с масками (217 Eyes-defy + 82 телефонных), шаг 1 анализа фото, 2026-10-31
LIMITS = {'wfrac_min': 0.423, 'aspect_max': 1.024, 'share_min': SHARE_MIN,
          'sclera_cx': (-0.082, 1.321), 'sclera_cy': (-1.004, 0.0)}


def parts(mask):
    """→ (метки, статистика cv2 по частям без фона, центры) 8-связности."""
    _, lab, st, cen = cv2.connectedComponentsWithStats(np.asarray(mask, np.uint8), 8)
    return lab, st[1:], cen[1:]


def strip_stats(conj):
    """Главная (крупнейшая) часть маски века → dict или None: доля площади, ширина (доля ширины кадра), высота/ширина,
    рамка x0, y0, w, h и центр."""
    lab, st, cen = parts(conj)
    if not len(st):
        return None
    b = int(np.argmax(st[:, cv2.CC_STAT_AREA]))
    x0, y0, w, h, a = (int(v) for v in st[b])
    return {'label': b + 1, 'share': a / st[:, cv2.CC_STAT_AREA].sum(), 'wfrac': w / conj.shape[1], 'aspect': h / w,
            'box': (x0, y0, w, h), 'center': (float(cen[b][0]), float(cen[b][1])), 'lab': lab}


def sclera_positions(sclera, strip):
    """Части склеры относительно полосы века → [(метка, доля площади склеры, cx, cy)]: cx — по горизонтали от левого
    края полосы в её ширинах (0…1 — над полосой), cy — центр части минус центр полосы, в ширинах (< 0 — выше)."""
    _, st, cen = parts(sclera)
    x0, _, w, _ = strip['box']
    tot = max(int(st[:, cv2.CC_STAT_AREA].sum()), 1) if len(st) else 1
    return [(i + 1, st[i, cv2.CC_STAT_AREA] / tot, (cen[i][0] - x0) / w, (cen[i][1] - strip['center'][1]) / w)
            for i in range(len(st))]


def image_stats(conj_gt, sclera):
    """Признаки одного обучающего снимка для fit_limits (маска врача на вырезке шага 1 + пороговая склера)."""
    s = strip_stats(conj_gt)
    if s is None:
        return None
    pos = [] if sclera is None or not np.any(sclera) else [p for p in sclera_positions(sclera, s) if p[1] >= PART_MIN]
    return {'wfrac': s['wfrac'], 'aspect': s['aspect'], 'sclera': [(cx, cy) for _, _, cx, cy in pos]}


def fit_limits(stats):
    """Пределы по обучающим снимкам (список image_stats, None пропускаются). Только маски — Hb не нужен."""
    stats = [s for s in stats if s is not None]
    if not stats:
        raise ValueError('нет обучающих масок для пределов анатомии')
    w = np.array([s['wfrac'] for s in stats])
    asp = np.array([s['aspect'] for s in stats])
    pos = np.array([p for s in stats for p in s['sclera']] or [(0.5, -0.5)])
    cx = np.percentile(pos[:, 0], [2, 98])
    cy = np.percentile(pos[:, 1], [2, 98])
    return {'wfrac_min': round(float(WFRAC_K * np.median(w)), 3),
            'aspect_max': round(float(ASPECT_K * np.percentile(asp, 98)), 3), 'share_min': SHARE_MIN,
            'sclera_cx': (round(float(cx[0] - SCLERA_PAD), 3), round(float(cx[1] + SCLERA_PAD), 3)),
            # склера — над полосой века: центр ниже центра полосы анатомически невозможен (под веком — кожа, насадка)
            'sclera_cy': (round(float(cy[0] - SCLERA_PAD), 3), min(0.0, round(float(cy[1] + SCLERA_PAD), 3)))}


def check(conj, sclera, limits=None):
    """Маски тканей одного снимка → dict:
        conj, sclera — маски после ограничений (конъюнктива — только главная полоса; склера — только части над ней);
        conj_ok — False: веко не найдено надёжно (анализ отклоняется целиком), conj_why — почему;
        sclera_ok — False: склера не используется, sclera_why; stats — числа для диагностики.
    Пустая конъюнктива возвращается как есть (conj_ok False без причины — её даёт вызывающий код)."""
    lim = LIMITS if limits is None else limits
    conj = np.asarray(conj, bool)
    sclera = np.zeros_like(conj) if sclera is None else np.asarray(sclera, bool)
    out = {'conj': conj, 'sclera': sclera, 'conj_ok': False, 'conj_why': '', 'sclera_ok': bool(sclera.any()),
           'sclera_why': '', 'stats': {}}
    s = strip_stats(conj)
    if s is None:
        out['sclera_ok'] = False
        return out
    st = {'полоса века: доля маски': round(s['share'], 2), 'ширина, доля кадра': round(s['wfrac'], 2),
          'высота/ширина': round(s['aspect'], 2)}
    out['stats'] = st
    why = []
    if s['share'] < lim['share_min']:
        why.append(f'маска века раздроблена: крупнейшая часть — {s["share"]:.0%} площади (нужно ≥ '
                   f'{lim["share_min"]:.0%}), остальное — кожа, пальцы или другие ткани')
    if s['wfrac'] < lim['wfrac_min']:
        why.append(f'полоса века занимает {s["wfrac"]:.0%} ширины кадра (у обучающих снимков ≥ {lim["wfrac_min"]:.0%}): '
                   'снимок не крупный план глаза или маска не на веке')
    if s['aspect'] > lim['aspect_max']:
        why.append(f'«веко» — пятно, а не полоса (высота/ширина {s["aspect"]:.2f}, у обучающих ≤ {lim["aspect_max"]:.2f})')
    main = s['lab'] == s['label']
    st['убрано частей вне полосы века, px'] = int(conj.sum() - main.sum())
    out['conj'] = main
    out['conj_ok'] = not why
    out['conj_why'] = '; '.join(why)
    out['sclera'] = sclera & ~main
    if not out['sclera'].any():
        out['sclera_ok'] = False
        return out
    keep = np.zeros_like(sclera)
    dropped = 0.0
    lab, _, _ = parts(out['sclera'])
    for i, a, cx, cy in sclera_positions(out['sclera'], s):
        if lim['sclera_cx'][0] <= cx <= lim['sclera_cx'][1] and lim['sclera_cy'][0] <= cy <= lim['sclera_cy'][1]:
            keep |= lab == i
        else:
            dropped += a
    st['склера: убрано вне глазного яблока, доля'] = round(dropped, 2)
    out['sclera'] = keep
    if dropped > SCLERA_DROP_MAX or not keep.any():
        out['sclera'] = np.zeros_like(sclera)
        out['sclera_ok'] = False
        out['sclera_why'] = (f'склера не над полосой века: {dropped:.0%} её площади вне глазного яблока (под веком, '
                             'сбоку от глаза или далеко над ним — кожа, пальцы, фон)')
    return out
