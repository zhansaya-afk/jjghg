"""Качество вырезки глаза по исходным пикселям (2026-11-01): сколько реальных пикселей у века, резкость без
увеличения, шум и блоки JPEG.

Почему отдельно от regions.quality: вырезка приводится к regions.WORK_SIDE (320 px) — мелкую увеличивают. Резкость
blur_ratio, измеренная после увеличения, падает вместе с коэффициентом увеличения: резкий, но мелкий глаз выглядел
«размытым», а увеличение не добавляет реальных деталей. Здесь всё меряется на исходной вырезке (пиксели фото):
    — резкость — без увеличения: мелкая вырезка как есть, крупная — уменьшенная до WORK_SIDE (как раньше), поэтому
      для крупных вырезок проверка размытия та же, что в regions.quality, и не слабее;
    — разрешение — ширина и толщина полосы века в пикселях исходного фото;
    — блочность JPEG (сетка 8×8 исходного фото): при артефактах сжатия деталей меньше, чем пикселей, поэтому порог
      ширины века строже; шум (оценка Immerkær) — только для диагностики: на коже он путается с текстурой;
    — размытие относительно размера века сверх неизбежного при дискретизации: расфокус мелкой вырезки.
Пороги (LIMITS) — fit_limits по синтетической деградации снимков обучающих фолдов Eyes-defy (уменьшение, JPEG, шум;
маски врачей и фолдовый сегментатор), без Hb и меток анемии. Увеличение вырезки нужно только сегментатору (он обучен
на 256–320 px) и никогда не считается улучшением качества.
Все функции чистые: изображение и маски на входе, числа на выходе.
"""
import cv2
import numpy as np

BLUR_MIN = 1.3            # тот же порог резкости, что в regions.quality (не ослабляется)
MEASURE_SIDE = 320        # = regions.WORK_SIDE: крупнее — уменьшить до него, мельче — мерить как есть (не увеличивать)
DRIFT_MAX = 0.10          # правило вывода порогов (задано до расчёта): медиана сдвига признаков цвета века ≤ 0,10 SD
IOU_DROP_MAX = 0.03       # … и средний IoU сегментатора падает не больше чем на 0,03 от полного разрешения

# fit_limits, 2026-11-01: медиана по 5 наборам обучающих фолдов Eyes-defy (216 снимков с масками врачей × 38 вариантов
# деградации; по фолдам: lid_w_clean 26–27, lid_w_artifact 105–107, block_max 1,369–1,436, sig_pix 0,927–0,936,
# rel_blur_max 0,0128–0,0132)
LIMITS = {'lid_w_clean': 27, 'lid_w_artifact': 106, 'block_max': 1.419, 'sig_pix': 0.928, 'rel_blur_max': 0.013,
          # ширина века по качеству JPEG файла (то же правило, чистое сжатие OpenCV 4:2:0 без шума, обучающие фолды;
          # по фолдам: 95 → 75 везде; 90 → 90–91; 85 → 90–105): сжатие с прореживанием цвета размывает цвет тонкой
          # полосы века даже при качестве 95. Ниже 85 — lid_w_artifact.
          'lid_w_jpeg': ((95, 75), (90, 90), (85, 105)),
          # файл не JPEG (PNG, скриншот, массив): история сжатия неизвестна — как лучший случай сжатия (JPEG 95)
          'lid_w_unknown': 75}

# Стандартная таблица квантования яркости JPEG (IJG, ITU-T T.81 K.1) — для оценки качества файла
STD_LUMA = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
                     49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], float)


def jpeg_quality(photo):
    """Путь / bytes / файловый объект / PIL.Image → качество JPEG файла 1–100 (оценка IJG по средней таблице
    квантования яркости; порядок элементов таблицы не важен) или None: не JPEG (PNG, скриншот, массив) — история
    сжатия неизвестна. Только чтение заголовка, пиксели не декодируются."""
    import io
    import os

    from PIL import Image
    try:
        if isinstance(photo, Image.Image):
            im = photo
        elif isinstance(photo, (bytes, bytearray)):
            im = Image.open(io.BytesIO(photo))
        elif isinstance(photo, (str, os.PathLike)):
            im = Image.open(photo)
        elif hasattr(photo, 'read') and hasattr(photo, 'seek'):
            pos = photo.tell()
            im = Image.open(io.BytesIO(photo.read()))
            photo.seek(pos)
        else:
            return None
        tables = getattr(im, 'quantization', None)
        if im.format != 'JPEG' or not tables:
            return None
        t = np.asarray(tables[0] if 0 in tables else next(iter(tables.values())), float)
    except Exception:  # noqa: BLE001 — заголовок не прочитан: качество неизвестно, проверка строже
        return None
    sc = 100 * t.mean() / STD_LUMA.mean()
    q = (200 - sc) / 2 if sc <= 100 else 5000 / sc
    return round(float(np.clip(q, 1, 100)))


def gray255(rgb):
    return cv2.cvtColor(np.clip(np.asarray(rgb, np.float32), 0, 1), cv2.COLOR_RGB2GRAY) * 255


def noise_sigma(rgb):
    """Шум, единицы яркости 0–255 (Immerkær 1996: свёртка с ядром второй разности, без порога по краям)."""
    g = gray255(rgb)
    h, w = g.shape
    if h < 3 or w < 3:
        return 0.0
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(g, cv2.CV_32F, k)[1:-1, 1:-1]
    return float(np.sqrt(np.pi / 2) * np.abs(r).mean() / 6)


def blockiness(rgb, offset=(0, 0)):
    """Блоки JPEG: скачок яркости на границах сетки 8×8 / скачок внутри блоков (≈ 1 — блоков нет). Скачок столбца
    (строки) — медиана по строкам (столбцам): отдельные края предметов на границе сетки не дают «блоков», а слабые
    ровные ступеньки сжатия на гладкой коже и слизистой дают. offset — (x0, y0) вырезки на исходном фото: сетка JPEG
    привязана к фото, а не к вырезке."""
    g = gray255(rgb)
    out = []
    for axis, off in ((1, offset[0]), (0, offset[1])):
        d = np.abs(np.diff(g, axis=axis))
        n = d.shape[axis]
        if n < 16 or d.shape[1 - axis] < 8:
            continue
        prof = np.median(d, axis=1 - axis)                      # по одному числу на столбец (строку)
        idx = (np.arange(n) + off + 1) % 8 == 0                 # скачок между пикселями 8k−1 и 8k фото
        out.append((prof[idx].mean() + 0.25) / (prof[~idx].mean() + 0.25))
    return float(np.mean(out)) if out else 1.0


def measure_scale(src_shape):
    """Масштаб измерения резкости: ≤ 1 (никогда не увеличиваем)."""
    return min(1.0, MEASURE_SIDE / max(src_shape[:2]))


def blur_ratio(rgb, roi):
    """Как regions.blur_ratio: |лапласиан| у границ ткани / то же после размытия σ = 1,5."""
    g = gray255(rgb) / 255
    band = cv2.dilate(np.asarray(roi, np.uint8), np.ones((9, 9), np.uint8)) > 0
    if not band.any():
        return float('nan')
    e0 = np.abs(cv2.Laplacian(g, cv2.CV_32F))[band].mean()
    e1 = np.abs(cv2.Laplacian(cv2.GaussianBlur(g, (0, 0), 1.5), cv2.CV_32F))[band].mean()
    return float(e0 / (e1 + 1e-6))


def strip_px(mask_src):
    """Полоса века в пикселях исходного фото → (ширина, толщина): ширина — по рамке главной части, толщина —
    медиана высоты маски по столбцам."""
    m = np.asarray(mask_src, bool)
    if not m.any():
        return 0, 0.0
    _, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    b = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    main = lab == b
    cols = main.sum(0)
    return int(st[b, cv2.CC_STAT_WIDTH]), float(np.median(cols[cols > 0]))


def to_source(mask, src_shape):
    """Маска вырезки (любого размера) → размер исходной вырезки."""
    return cv2.resize(np.asarray(mask, np.uint8), (src_shape[1], src_shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def measure(src, lid, roi=None, offset=(0, 0)):
    """Исходная вырезка (пиксели фото, float RGB 0–1) + маски на вырезке анализа (любого размера) → числа качества:
    lid_w, lid_h — полоса века lid в px исходника; blur — резкость у краёв roi (по умолчанию — века) на исходных
    пикселях без увеличения (крупная вырезка — уменьшенная до MEASURE_SIDE); noise, block — артефакты исходника."""
    m = to_source(lid, src.shape)
    r = m if roi is None else to_source(roi, src.shape)
    k = measure_scale(src.shape)
    img = src if k == 1 else cv2.resize(np.asarray(src, np.float32), (max(1, round(src.shape[1] * k)),
                                                                        max(1, round(src.shape[0] * k))),
                                         interpolation=cv2.INTER_AREA)
    rr = r if k == 1 else to_source(r, img.shape)
    w, t = strip_px(m)
    return {'lid_w': w, 'lid_h': round(t, 1), 'blur': blur_ratio(img, rr), 'measure_scale': round(k, 3),
            'noise': round(noise_sigma(src), 2), 'block': round(blockiness(src, offset), 3)}


def blur_sigma(blur):
    """Оценка σ размытия в пикселях измерения по blur_ratio: у гауссова края отношение = 1 + 1,5²/σ²."""
    return float(1.5 / np.sqrt(max(blur - 1, 1e-3)))


def rel_blur(q, limits=None):
    """Размытие сверх неизбежного при дискретизации (sig_pix — σ у резких уменьшенных снимков), в долях ширины
    века: мелкая, но резкая вырезка → ≈ 0 (её решает порог пикселей), расфокус → больше."""
    lim = LIMITS if limits is None else limits
    if q['lid_w'] <= 0 or not np.isfinite(q['blur']):
        return float('inf')
    sig = blur_sigma(q['blur'])
    excess = np.sqrt(max(sig ** 2 - lim['sig_pix'] ** 2, 0.0))
    return float(excess / (q['lid_w'] * q['measure_scale']))


def artifacts(q, limits=None):
    lim = LIMITS if limits is None else limits
    return q['block'] > lim['block_max']


def lid_needed(q, limits=None, file_q=None):
    """Наименьшая ширина века (px исходника) → (px, почему). file_q — качество JPEG файла (jpeg_quality), None —
    не JPEG / неизвестно, 'lossless' — источник без сжатия (только если это известно наверняка). Видимые блоки JPEG
    в вырезке — не мягче lid_w_artifact. Выбирается более строгое."""
    lim = LIMITS if limits is None else limits
    if file_q == 'lossless':
        need, why = lim['lid_w_clean'], 'без сжатия'
    elif file_q is None:
        need, why = lim['lid_w_unknown'], 'файл не JPEG или сжатие неизвестно'
    else:
        need = next((w for qq, w in lim['lid_w_jpeg'] if file_q >= qq), lim['lid_w_artifact'])
        why = f'JPEG качества {file_q}'
    if artifacts(q, lim) and lim['lid_w_artifact'] > need:
        need, why = lim['lid_w_artifact'], f'видны блоки JPEG {q["block"]:.2f}'
    return need, why


def check(q, limits=None, file_q=None):
    """Числа measure → список проблем (пусто — исходных деталей достаточно). Отдельно: «мало пикселей» и «размыто».
    file_q — см. lid_needed."""
    lim = LIMITS if limits is None else limits
    issues = []
    need, why = lid_needed(q, lim, file_q)
    if q['lid_w'] < need:
        issues.append(f'мало реальных пикселей: веко {q["lid_w"]} px на исходном фото, при сжатии «{why}» нужно ≥ '
                      f'{need} px (сжатие размывает цвет тонкой полосы века) — снимите глаз ближе или пришлите '
                      'оригинал фото, не скриншот и не пересланное через мессенджер')
    if not np.isnan(q['blur']) and q['blur'] < BLUR_MIN:
        issues.append(f'снимок размыт (резкость {q["blur"]:.2f} < {BLUR_MIN} на исходных пикселях, без увеличения)')
    elif rel_blur(q, lim) > lim['rel_blur_max']:
        issues.append(f'снимок размыт относительно размера глаза (размытие {rel_blur(q, lim):.3f} ширины века, '
                      f'допустимо ≤ {lim["rel_blur_max"]:.3f}) — увеличение вырезки резкость не вернёт')
    return issues


def _first_good(s, ascending):
    """s — группы (индекс — значение признака, колонки drift, iou_drop, v); от лучших групп к худшим до первой,
    где правило DRIFT_MAX / IOU_DROP_MAX не выполняется → граничное значение v или None."""
    s = s.sort_index(ascending=ascending)
    good = ((s.drift <= DRIFT_MAX) & (s.iou_drop <= IOU_DROP_MAX)).cumprod().astype(bool)
    if not good.any():
        return None
    return s.v[good].max() if ascending else s.v[good].min()


def fit_limits(rows, min_n=20):
    """Пороги по таблице синтетической деградации обучающих снимков (Hb и метки не нужны): rows — dict с полями
    kind ('clean' / 'artifact' / 'defocus'), lid_w, drift (медианный сдвиг признаков цвета века, SD), iou_drop,
    block, blur, measure_scale. Правило DRIFT_MAX / IOU_DROP_MAX задано заранее. Ширина века — наименьшая, начиная с которой
    (и для всех более широких групп) правило выполняется; группы — по log2 ширины с шагом 1/4, медиана сдвига,
    средняя потеря IoU. Блоки JPEG — 99-й перцентиль чистых снимков. Относительное размытие — наибольшее, до
    которого правило выполняется (снимки clean и defocus с веком не уже порога чистых)."""
    import pandas as pd
    T = pd.DataFrame(rows)
    out = {}

    def groups(g, col, step):
        b = (np.log2(g[col].clip(lower=1e-6)) / step).round() * step
        s = g.assign(b=b).groupby('b').agg(drift=('drift', 'median'), iou_drop=('iou_drop', 'mean'), v=(col, 'median'),
                                           n=('drift', 'size'))
        return s[s.n >= min_n]

    for kind in ('clean', 'artifact'):
        v = _first_good(groups(T[T.kind == kind], 'lid_w', 0.25), ascending=False)
        out['lid_w_' + kind] = int(np.ceil(v)) if v is not None else int(T.lid_w.max())
    out['block_max'] = round(float(np.percentile(T[T.kind == 'clean'].block, 99)), 3)
    clean = T[T.kind == 'clean']
    out['sig_pix'] = round(float(np.percentile([blur_sigma(b) for b in clean.blur], 90)), 3)
    T = T.assign(rel_blur=[rel_blur(r, out) for r in T[['lid_w', 'blur', 'measure_scale']].to_dict('records')])
    g = T[T.kind.isin(['clean', 'defocus']) & (T.lid_w >= out['lid_w_clean']) & np.isfinite(T.rel_blur)]
    v = _first_good(groups(g, 'rel_blur', 0.25), ascending=True)
    out['rel_blur_max'] = float(np.ceil(v * 1e4) / 1e4) if v is not None else 0.0
    return out
