"""Одно фото → оценка риска анемии. Главная функция — analyze_photo; модель обучает scripts/train_core.py.

    from anemia.predict import analyze_photo, format_result
    r = analyze_photo('eye.jpg', sex='F', age=30)          # путь, байты, PIL.Image или массив RGB
    print(format_result(r))

Шаги (те же, что при обучении, иначе признаки не совпадут): чтение с поворотом по EXIF → проверка кадра →
поиск конъюнктивы и склеры (anemia.regions, без обучения, GrabCut с фиксированным сидом) → контроль качества
тканей → признаки цвета v12 → проверка «снимок похож на обучающие» (OOD) → P(анемия) → уровень риска.
Непригодный или нетипичный снимок → status ≠ 'ok', и ни уровня риска, ни цифры Hb не выдаётся.
Диагноз не ставится; Hb — экспериментальный ориентир, только если модель Hb на CV заметно лучше константы.
Лабораторный Hb сюда не передаётся и с фото не смешивается: для него anemia.labs.lab_hb_report.
"""
import io
import os

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from . import core as CORE
from . import v12_model as V12
from .io import resize_max
from .regions import detect_regions, quality

MODEL_FILE = 'hemavision_core.joblib'
MIN_SIDE = 24            # px по короткой стороне (вырезы CP-AnemiC бывают 171×34); мелкую ткань ловит quality()
STATUS_RU = {'ok': 'оценка выполнена', 'retake': 'переснимите фото', 'unreliable': 'оценка ненадёжна'}
RISK_TEXT = {'высокий': 'признаки бледности выражены — сдайте общий анализ крови (Hb)',
             'средний': 'по фото уверенно сказать нельзя — сдайте общий анализ крови (Hb)',
             'низкий': 'признаков анемии по фото нет; это не исключает анемию, при симптомах сдайте анализ крови'}
DISCLAIMER = 'Скрининг риска по фото, не диагноз. Решение принимает врач по анализу крови.'


class PhotoError(ValueError):
    """Файл нельзя использовать как фото глаза (не открылся, слишком мал, чёрно-белый...)."""


# ------------------------------------------------------------------ чтение
def load_photo(src):
    """Путь / bytes / файловый объект / PIL.Image / массив → float32 RGB [0, 1] исходного размера.
    Как anemia.io.load_rgb (поворот по EXIF, 8 бит), но прозрачный фон (вырезанная ткань) становится чёрным,
    16-битные и float-массивы приводятся к [0, 1], а ошибки — понятным PhotoError."""
    if isinstance(src, np.ndarray):
        a = src
        if a.ndim == 2 or (a.ndim == 3 and a.shape[2] == 1):
            raise PhotoError('чёрно-белое изображение: нужно цветное фото')
        if a.ndim != 3 or a.shape[2] not in (3, 4):
            raise PhotoError(f'ожидается массив H×W×3 (RGB), получено {a.shape}')
        if a.dtype == np.uint8:
            a = a.astype(np.float32) / 255
        elif a.dtype == np.uint16:
            a = a.astype(np.float32) / 65535
        else:
            a = a.astype(np.float32)
            if np.nanmax(a) > 1.5:
                a = a / 255
        if a.shape[2] == 4:
            a = np.where(a[..., 3:] <= 0, 0, a[..., :3])
        return np.ascontiguousarray(np.clip(np.nan_to_num(a), 0, 1), np.float32)
    if isinstance(src, (str, os.PathLike)):
        if not os.path.isfile(src):
            raise PhotoError(f'файл не найден: {src}')
        if os.path.getsize(src) == 0:
            raise PhotoError(f'пустой файл: {src}')
        if str(src).lower().endswith(('.heic', '.heif')):
            try:
                from pillow_heif import register_heif_opener
                register_heif_opener()
            except ImportError:
                raise PhotoError('HEIC (iPhone): pip install pillow-heif или сохраните фото как JPEG') from None
    elif isinstance(src, (bytes, bytearray)):
        src = io.BytesIO(src)
    try:
        im = src if isinstance(src, Image.Image) else Image.open(src)
        im.load()
    except (UnidentifiedImageError, OSError) as e:
        raise PhotoError(f'файл не открывается как изображение ({type(e).__name__}): {str(e)[:120]}') from None
    im = ImageOps.exif_transpose(im)
    if im.mode in ('L', 'LA', '1', 'I', 'I;16', 'F'):
        raise PhotoError('чёрно-белое изображение: нужно цветное фото')
    alpha = None
    if im.mode in ('RGBA', 'LA', 'PA') or (im.mode == 'P' and 'transparency' in im.info):
        alpha = np.asarray(im.convert('RGBA'))[..., 3]
    rgb = np.asarray(im.convert('RGB'), np.float32) / 255
    if alpha is not None:
        rgb[alpha == 0] = 0                        # как в CP-AnemiC: прозрачный фон — чёрный
    return rgb


def frame_issues(rgb):
    """Проблемы всего кадра, при которых ткани искать бессмысленно → список строк (пустой — кадр годен)."""
    h, w = rgb.shape[:2]
    if min(h, w) < MIN_SIDE:
        return [f'изображение слишком маленькое ({w}×{h}); нужно хотя бы {MIN_SIDE} px по короткой стороне']
    small = resize_max(rgb, 256)
    v = small.max(-1)
    issues = []
    if np.percentile(np.abs(small[..., 0] - small[..., 1]) + np.abs(small[..., 1] - small[..., 2]), 95) < 0.02:
        issues.append('изображение почти без цвета (чёрно-белое или сильный фильтр)')
    if np.percentile(v, 99) < 0.15:
        issues.append('кадр почти чёрный — включите свет')
    if (small.min(-1) > 0.98).mean() > 0.5:
        issues.append('кадр пересвечен — уберите вспышку или прямой свет')
    return issues


# ------------------------------------------------------------------ модель
_BUNDLES = {}


def default_model_path():
    base = os.environ.get('ANEMIA_MODELS') or os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    return os.path.join(base, MODEL_FILE)


def load_bundle(path=None):
    """Модель из scripts/train_core.py (кэшируется в памяти)."""
    path = path or default_model_path()
    if path not in _BUNDLES:
        if not os.path.exists(path):
            raise FileNotFoundError(f'нет модели {path}: сначала python scripts/train_core.py')
        import joblib
        _BUNDLES[path] = joblib.load(path)
    return _BUNDLES[path]


def _sex(sex):
    if sex is None:
        return None
    s = str(sex).strip().upper()
    return 'M' if s[:1] in ('M', 'М') else 'F' if s[:1] in ('F', 'Ж', 'W') else None


def pick_model(bundle, age):
    """Дети до 5 лет — модель CP-AnemiC, остальные — модель взрослых (Eyes-defy)."""
    child = age is not None and age < 5
    key = 'cp_anemic' if child else 'eyes_defy'
    if key not in bundle['models']:
        raise KeyError(f'в модели нет части «{key}» (обучена на {sorted(bundle["models"])})')
    return key


# ------------------------------------------------------------------ главный вход
def _detect(rgb):
    img = resize_max(rgb, V12.DETECT_SIDE)
    cv2.setRNGSeed(0)                       # GrabCut случаен: тот же сид, что при обучении (evaluate._v12_compute)
    return detect_regions(img, 'both')


def analyze_photo(photo, sex=None, age=None, pregnant=False, bundle=None):
    """Фото глаза (оттянутое нижнее веко; можно уже вырезанную конъюнктиву) → dict:
        status   — 'ok' | 'retake' (снимок непригоден) | 'unreliable' (снимок не похож на обучающие);
        risk     — 'низкий' / 'средний' / 'высокий' или None; p_anemia — P(анемия) модели или None;
        hb_range — (от, до) г/л, экспериментально, или None; hb_note — почему цифры Hb нет / что она значит;
        model, tissues, kind — чем и по какой ткани оценено; reasons — почему такой ответ; issues — проблемы снимка;
        crop, conj, sclera — вырезка и маски (для показа, anemia.predict.overlay)."""
    bundle = bundle if bundle is not None else load_bundle()
    sex = _sex(sex)
    key = pick_model(bundle, age)
    B = bundle['models'][key]
    r = {'status': 'retake', 'risk': None, 'p_anemia': None, 'hb_range': None, 'hb_note': None, 'model': B['title'],
         'features': None,
         'tissues': [], 'kind': None, 'reasons': [], 'issues': [], 'notes': [], 'crop': None, 'conj': None,
         'sclera': None, 'disclaimer': DISCLAIMER}
    try:
        rgb = load_photo(photo)
    except PhotoError as e:
        r['issues'] = [str(e)]
        return r
    r['issues'] = frame_issues(rgb)
    if r['issues']:
        return r
    det = _detect(rgb)                      # ориентация — как есть (EXIF учтён); повёрнутый кадр меняет ответ
    crop, conj, sclera = det['crop'], det['conj'], det['sclera']
    r.update(crop=crop, conj=conj, sclera=sclera, kind=det['kind'], tissues=sorted(det['use']),
             notes=list(det['notes']))
    r['issues'] = quality(crop, conj, sclera, det['src_side'])
    if 'conj' not in det['use']:
        r['issues'].append('конъюнктива не найдена — оттяните нижнее веко и снимите его крупно')
    if r['issues']:
        return r
    if key == 'eyes_defy' and det['kind'] == 'вырезанная ткань':
        r['status'] = 'unreliable'
        r['issues'].append('это уже вырезанная ткань, а модель взрослых обучена на снимках глаза целиком — '
                           'пришлите фото глаза с оттянутым нижним веком')
        return r
    if key == 'cp_anemic' and det['kind'] == 'вырезанная ткань':
        crop = resize_max(rgb, V12.CROP_SIDE)            # как обучались вырезы CP-AnemiC: весь кадр — ткань
        conj, sclera = np.ones(crop.shape[:2], bool), None
    elif key == 'cp_anemic':
        r['status'] = 'unreliable'
        r['issues'].append('модель детей до 5 лет обучена только на вырезанной конъюнктиве (CP-AnemiC); на снимке '
                           'глаза целиком она не проверялась — пришлите вырез конъюнктивы или сдайте анализ крови')
        return r
    f = V12.eye_features(crop, conj, sclera if sclera is not None and sclera.any() else None)
    f['male'] = 1.0 if sex == 'M' else 0.0 if sex == 'F' else 0.5
    x = np.array([f.get(k, np.nan) for k in B['feats']], float)
    r['features'] = dict(zip(B['feats'], x.tolist()))
    if not np.isfinite(x[[i for i, k in enumerate(B['feats']) if k.startswith('c_')]]).all():
        r['issues'].append('мало чистой ткани конъюнктивы для признаков цвета (блики, тени, ресницы)')
        return r
    p, lv, hb, ood = (v[0] for v in CORE.apply_core(B, x[None]))
    if ood > CORE.OOD_MAX:
        r['status'] = 'unreliable'
        r['issues'].append(f'{ood * 100:.0f}% признаков цвета далеко вне диапазона обучающих снимков '
                           '(другой свет, камера или не та ткань) — цифрам не верим, переснимите при дневном свете')
        return r
    r['status'], r['p_anemia'] = 'ok', float(p)
    r['risk'] = CORE.RISK_LEVELS[int(lv)]
    has_s = 's_EI_m' in B['feats'] and np.isfinite(f.get('s_EI_m', np.nan))
    r['reasons'].append('оценено по цвету конъюнктивы' + (
        ' и склеры; цвет века дополнительно нормирован по белку глаза' if has_s else
        '' if key == 'cp_anemic' else ' (склера не найдена — без поправки на свет по белку глаза)'))
    low = (f'≤ {B["t_lo"]:.2f} — низкий' if np.isfinite(B['t_lo']) else
           'уровня «низкий» у этой модели нет (на обучающих пациентах не нашлось зоны, где здоровых '
           f'≥ {B["zone_target"]:.0%})')
    high = (f'≥ {B["t_hi"]:.2f} — высокий' if np.isfinite(B['t_hi']) else 'уровня «высокий» у этой модели нет')
    r['reasons'].append(f'P(анемия) = {p:.2f}; пороги модели: {low}, {high} (подобраны вложенной CV на обучающих пациентах '
                        f'с целью ≥ {B["zone_target"]:.0%} верных; на невиданных пациентах точность ниже — см. evaluate)')
    if B['hb_ok']:
        hw = B['hb_halfwidth']
        lo, hi = 5 * np.floor((hb - hw) / 5), 5 * np.ceil((hb + hw) / 5)
        r['hb_range'] = (float(lo), float(hi))
        r['hb_note'] = (f'экспериментально: на вложенной CV обучающих пациентов в 80% случаев реальный Hb был '
                        f'в пределах ±{hw:.0f} г/л от прогноза (MAE {B["hb_mae_cv"]:.0f} г/л против {B["hb_mae_const_cv"]:.0f} у константы)')
    else:
        r['hb_note'] = (f'цифра Hb не выдаётся: на проверке модель Hb почти не лучше константы '
                        f'(MAE {B["hb_mae_cv"]:.0f} против {B["hb_mae_const_cv"]:.0f} г/л)')
    if pregnant or (age is not None and 5 <= age < 18):
        r['notes'].append('модель обучена на небеременных взрослых: для беременных и подростков оценка ориентировочная')
    return r


def format_result(r):
    """Текст результата для печати."""
    lines = [f"Статус: {STATUS_RU[r['status']]}", f"Модель: {r['model']}"]
    if r['kind']:
        lines.append(f"На снимке: {r['kind']}; ткани: {', '.join(r['tissues']) or 'не найдены'}")
    if r['status'] == 'ok':
        lines.append(f"РИСК АНЕМИИ: {r['risk'].upper()} — {RISK_TEXT[r['risk']]}")
        if r['hb_range']:
            lines.append(f"Ориентировочный Hb: {r['hb_range'][0]:.0f}–{r['hb_range'][1]:.0f} г/л ({r['hb_note']})")
        else:
            lines.append(f"Hb: {r['hb_note']}")
        lines += [f'Почему: {x}' for x in r['reasons']]
    else:
        lines += [f'Проблема: {x}' for x in r['issues']]
    lines += [f'Примечание: {x}' for x in r['notes']]
    lines.append(r['disclaimer'])
    return '\n'.join(lines)


def overlay(r):
    """Вырезка с контурами: зелёный — конъюнктива, голубой — склера (uint8 RGB) или None."""
    if r['crop'] is None:
        return None
    vis = (np.clip(r['crop'], 0, 1) * 255).astype(np.uint8).copy()
    for m, col in ((r['conj'], (0, 255, 0)), (r['sclera'], (0, 160, 255))):
        if m is not None and m.shape == vis.shape[:2] and m.any():
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(vis, cnts, -1, col, 2)
    return vis
