"""Сегментация конъюнктивы = подход v11: попиксельный HistGradientBoosting по цветовым и текстурным признакам
+ положение относительно радужки, обучение на масках врачей Eyes-defy-anemia. Перенесено из md/current/v11_cells/
cell2–3. Точка отсчёта для этапа 2 (обученная сеть) и — с 2026-10-18 — маска конъюнктивы ядра (conj_mask):
пороговый поиск anemia.regions захватывал вместе с веком склеру, кожу и пальцы (IoU с масками врачей ≈ 0,30 —
разовый замер 2026-10-18, не evaluate; у сегментатора v11 в evaluate baseline_v11 средний IoU 0,58).
"""
import cv2
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from .color import lin_to_srgb, srgb_to_lin
from .io import resize_max, resize_to

SEG_SIZE = 256
SIGMAS = (0, 2, 4, 8, 16)
PIX_PER_IMAGE = 900


def normalize_light(rgb):
    """«Серый мир» (shades of gray, p=6) + выравнивание яркости — только для признаков сегментации."""
    lin = srgb_to_lin(np.clip(rgb, 0, 1)).reshape(-1, 3)
    g = (np.mean(lin ** 6, axis=0) + 1e-12) ** (1 / 6)
    lin = lin * (g.mean() / g)
    lin = lin * (0.8 / max(np.percentile(lin.max(1), 95), 1e-3))
    return lin_to_srgb(lin).reshape(rgb.shape)


def find_iris(rgb):
    """Самое тёмное компактное округлое пятно ближе к центру → (cx, cy, r) или None."""
    L = cv2.cvtColor(np.clip(rgb, 0, 1).astype(np.float32), cv2.COLOR_RGB2Lab)[..., 0]
    L = cv2.GaussianBlur(L, (0, 0), 2)
    h, w = L.shape
    best = None
    for q in (6, 10, 15):
        n, _, st, cen = cv2.connectedComponentsWithStats((L < np.percentile(L, q)).astype(np.uint8), 8)
        for i in range(1, n):
            x, y, bw, bh, a = st[i]
            if a < 0.004 * h * w or bw > 0.5 * w or bh > 0.6 * h:
                continue
            fill, aspect = a / (bw * bh), min(bw, bh) / max(bw, bh)
            cx, cy = cen[i]
            central = 1 - min(1, np.hypot((cx - w / 2) / w, (cy - h / 2) / h) * 1.6)
            score = fill * aspect * central * np.sqrt(a)
            if fill > 0.5 and aspect > 0.55 and (best is None or score > best[0]):
                best = (score, cx, cy, max(bw, bh) / 2)
        if best:
            break
    return None if best is None else best[1:]


def pixel_features(rgb):
    n = normalize_light(rgb)
    lab = cv2.cvtColor(n, cv2.COLOR_RGB2Lab)
    hsv = cv2.cvtColor(n, cv2.COLOR_RGB2HSV)
    c = np.clip(n, 1 / 255, 1)
    base = [lab[..., 0] / 100, lab[..., 1] / 50, lab[..., 2] / 50, hsv[..., 1],
            np.log(c[..., 0] / c[..., 1]), np.log(c[..., 1] / c[..., 2])]
    feats = [ch if s == 0 else cv2.GaussianBlur(ch, (0, 0), s) for ch in base for s in SIGMAS]
    Lb = lab[..., 0] / 100
    for s in (2, 4):
        b = cv2.GaussianBlur(Lb, (0, 0), s)
        feats.append(np.hypot(cv2.Sobel(b, cv2.CV_32F, 1, 0), cv2.Sobel(b, cv2.CV_32F, 0, 1)))
    a = lab[..., 1] / 50
    feats.append(np.sqrt(np.maximum(cv2.GaussianBlur(a * a, (0, 0), 5) - cv2.GaussianBlur(a, (0, 0), 5) ** 2, 0)))
    h, w = Lb.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    feats += [xx / w, yy / h]
    iris = find_iris(rgb)
    cx, cy, r = iris if iris is not None else (w / 2, h * 0.4, 0.12 * w)
    feats += [np.clip((xx - cx) / r, -6, 6), np.clip((yy - cy) / r, -6, 6)]
    return np.stack(feats, -1).astype(np.float32)


def _sample(F, mask, rng):
    ring = cv2.dilate(mask.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool) & ~mask
    pools = [np.flatnonzero(mask), np.flatnonzero(ring), np.flatnonzero(~mask & ~ring)]
    n = PIX_PER_IMAGE // 3
    idx = np.concatenate([rng.choice(p, min(n, len(p)), replace=False) for p in pools if len(p)])
    return F.reshape(-1, F.shape[-1])[idx], mask.reshape(-1)[idx].astype(int)


def train(feats, masks, seed=0):
    rng = np.random.default_rng(seed)
    XY = [_sample(F, m, rng) for F, m in zip(feats, masks)]
    X, y = np.concatenate([a for a, _ in XY]), np.concatenate([b for _, b in XY])
    return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=150, random_state=seed).fit(X, y)


def _largest_parts(mask, min_frac=0.25):
    n, lab, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if n <= 1:
        return mask
    areas = st[1:, cv2.CC_STAT_AREA]
    return np.isin(lab, 1 + np.flatnonzero(areas >= areas.max() * min_frac))


def _raw_mask(model, F, thr):
    h, w = F.shape[:2]
    p = model.predict_proba(F.reshape(-1, F.shape[-1]))[:, 1].reshape(h, w).astype(np.float32)
    m = cv2.GaussianBlur(p, (0, 0), 1.5) > thr
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0


def _main_parts(m):
    m = _largest_parts(m)
    return m if m.sum() >= 0.004 * m.size else np.zeros_like(m)


def predict(model, F, thr=0.5):
    return _main_parts(_raw_mask(model, F, thr))


def image_samples(rgb, mask, seed):
    """Обучающие пиксели одного снимка с маской врача → (X, y); свой сид у снимка — результат можно кэшировать."""
    return _sample(pixel_features(rgb), mask, np.random.default_rng(seed))


def fit_samples(samples, seed=0):
    """Сегментатор по пикселям image_samples нескольких снимков (те же настройки, что у train)."""
    X, y = np.concatenate([a for a, _ in samples]), np.concatenate([b for _, b in samples])
    return HistGradientBoostingClassifier(max_depth=6, learning_rate=0.1, max_iter=150, random_state=seed).fit(X, y)


def conj_mask(model, crop):
    """Маска конъюнктивы на вырезке любого размера → bool размера вырезки. Признаки пикселей — на SEG_SIZE,
    как при обучении; пусто — сегментатор конъюнктивы не нашёл. Части, касающиеся нижнего края вырезки,
    отбрасываются: вывернутое веко лежит над кожей века и пальцами и до низа кадра не доходит (Eyes-defy: 0 из 211
    снимков), а у низа бывают ногти, губы, пальцы."""
    m = _raw_mask(model, pixel_features(resize_max(np.ascontiguousarray(crop, np.float32), SEG_SIZE)), 0.5)
    _, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    bottom = np.unique(lab[-1])
    m &= ~np.isin(lab, bottom[bottom > 0])       # до выбора крупных частей: иначе пальцы у края «съедают» веко
    return resize_to(_main_parts(m), crop.shape[:2], nearest=True)
