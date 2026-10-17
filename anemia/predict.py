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
from . import regions as R
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


# ------------------------------------------------------------------ где глаз на снимке
FACE_MODEL_FILE = 'face_landmarker.task'
FACE_MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/'
                  'face_landmarker.task')
EYE_SMALL = 0.075        # радиус радужки меньше этой доли короткой стороны кадра — глаз мелкий, вырезаем его
IRIS_MIN_R = 0.012       # самая мелкая радужка, которую ищем (доля короткой стороны кадра 1024 px → ≥ 6 px)
ZOOM_MAX_AREA = 0.5     # рамка вокруг радужки — меньше половины кадра, иначе это уже крупный план
IRIS_CONTRAST = 20       # мелкую радужку принимаем, только если её край в 2 раза контрастнее, чем при обучении (10)
_FACE = {}


def face_landmarker(path=None, download=True):
    """MediaPipe FaceLandmarker (как в ноутбуке v12) → (детектор или None, пояснение). Модель лица (3,7 МБ)
    скачивается один раз рядом с моделью риска ($ANEMIA_MODELS). Без mediapipe или без сети — None: тогда
    глаз ищется по всему кадру и по радужке."""
    path = path or os.path.join(os.path.dirname(default_model_path()), FACE_MODEL_FILE)
    if path in _FACE:
        return _FACE[path]
    try:
        from mediapipe.tasks.python import BaseOptions, vision
    except Exception as e:
        _FACE[path] = (None, f'MediaPipe не установлен ({type(e).__name__}): лицо целиком не разобрать')
        return _FACE[path]
    try:
        if not os.path.exists(path):
            if not download:
                raise FileNotFoundError(path)
            import urllib.request
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with urllib.request.urlopen(FACE_MODEL_URL, timeout=30) as resp, open(path + '.part', 'wb') as out:
                out.write(resp.read())
            os.replace(path + '.part', path)
        opt = vision.FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=path), num_faces=1,
                                           min_face_detection_confidence=0.3, min_face_presence_confidence=0.3)
        _FACE[path] = (vision.FaceLandmarker.create_from_options(opt), 'MediaPipe: поиск глаза на лице включён')
    except Exception as e:
        _FACE[path] = (None, f'MediaPipe не запустился ({type(e).__name__}: {str(e)[:80]}): лицо целиком не разобрать')
    return _FACE[path]


def _analyze_box(rgb, scale, box, iris, kind):
    """Ткани внутри рамки глаза. Рамка и радужка — в координатах кадра DETECT_SIDE; вырезается из исходного фото
    (полное разрешение), пороги площади тканей — от рамки глаза, а не от всего кадра."""
    x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
    crop = R._prepare(rgb[y0:y1, x0:x1])
    s = crop.shape[1] / (x1 - x0)
    ir = None if iris is None else ((iris[0] * scale - x0) * s, (iris[1] * scale - y0) * s, iris[2] * scale * s)
    cv2.setRNGSeed(0)
    r = R._analyze_crop(crop, True, True, iris=ir, kind=kind)
    r['src_side'] = max(x1 - x0, y1 - y0)
    return r


def _iris_box(iris, H, W):
    """Рамка глаза вокруг радужки (ширина глаза ≈ 2,2 диаметра радужки; пропорции как у face_eye_crops)."""
    cx, cy, ri = iris
    e = 4.4 * ri
    return (max(0, cx - 0.9 * e), max(0, cy - 0.6 * e), min(W, cx + 0.9 * e), min(H, cy + 1.0 * e))


def _zoom(rgb, scale, det, iris, H, W, how):
    """Разобрать рамку вокруг радужки. Принимается, только если рядом с радужкой нашлась склера (иначе тёмный
    круг мог быть чем угодно: логотип, ноздря) и пригодных тканей больше, чем во всём кадре."""
    box = _iris_box(iris, H, W)
    if box[2] - box[0] < 20 or box[3] - box[1] < 20 or (box[2] - box[0]) * (box[3] - box[1]) > ZOOM_MAX_AREA * H * W:
        return None
    z = _analyze_box(rgb, scale, box, iris, 'глаз в части кадра')
    if 'sclera' in z['use'] and len(z['use']) > len(det['use']):
        z['eye'] = how
        return z
    return None


def _detect(rgb, landmarker=None):
    """Глаз по шагам → результат regions (crop, conj, sclera, use, notes, kind, src_side) + eye — как найден глаз
    (None — не найден).
    1) Весь кадр — ровно как при обучении (все снимки Eyes-defy проходят этим путём, признаки совпадают).
       Обе ткани нашлись или это вырезанная ткань → готово.
    2) Иначе MediaPipe ищет лицо; нашёл и глаз не во весь кадр → рамки глаз по точкам лица. Если в кадре была
       крупная радужка, рамки берутся, только если в них нашлась склера и пригодных тканей больше (на крупных
       планах MediaPipe иногда «видит» лицо, которого нет).
    3) Крупная радужка была → результат шага 1 (радужку, найденную во всём кадре, для приближения не используем:
       на крупных планах ею бывает зрачок).
    4) Ни радужки, ни лица → ищем мелкую радужку с надёжной (вдвое контрастнее обычного) границей и разбираем рамку
       вокруг неё; берём, только если рядом нашлась склера и пригодных тканей больше, чем во всём кадре."""
    img = resize_max(rgb, V12.DETECT_SIDE)
    H, W = img.shape[:2]
    scale = rgb.shape[1] / W
    cv2.setRNGSeed(0)                       # GrabCut случаен: тот же сид, что при обучении (evaluate._v12_compute)
    det = detect_regions(img, 'both')
    det['eye'] = 'глаз крупным планом' if det['iris'] is not None else 'весь кадр' if det['use'] else None
    if det['use'] == {'conj', 'sclera'} or det['kind'] == 'вырезанная ткань':
        return det
    faces = R.face_eye_crops(img, landmarker) if landmarker else []
    if faces and max(c['box'][2] - c['box'][0] for c in faces) <= 0.5 * W:
        best = None
        for c in faces:
            r = _analyze_box(rgb, scale, c['box'], c['iris'], c['how'])
            if best is None or r['score'] > best['score']:
                best = r
        best['eye'] = 'лицо в кадре, глаз по точкам MediaPipe'
        if det['iris'] is None or ('sclera' in best['use'] and len(best['use']) > len(det['use'])):
            return best                      # лицо целиком: остальной кадр не разбираем (губы, нос — не ткани)
    if det['iris'] is not None:
        return det                           # крупная радужка есть: снимок как обучающие, ничего не угадываем
    iris, con = R.find_iris(img, min_r=IRIS_MIN_R, max_r=EYE_SMALL, min_contrast=IRIS_CONTRAST, with_score=True)
    if iris is None or iris[2] > EYE_SMALL * min(H, W):     # тёмное пятно крупнее «мелкой радужки» — не она
        return det
    z = _zoom(rgb, scale, det, iris, H, W, f'мелкая радужка r={iris[2] * scale:.0f} px, контраст края {con:.0f}')
    return z or det


def tissue_status(crop, det):
    """Статус каждой ткани отдельно: 'пригодна' / 'не найдена: …' / 'плохое качество: …'."""
    out = {}
    for t, name in (('conj', 'конъюнктива'), ('sclera', 'склера')):
        m = det[t]
        if t in det['use'] and m is not None and m.any():
            iss = quality(crop, m, np.zeros_like(m), det['src_side'])
            out[t] = 'пригодна' if not iss else 'плохое качество: ' + '; '.join(iss)
        else:
            why = [n for n in det['notes'] if ('склер' in n) == (t == 'sclera') and 'одной ткани' not in n]
            out[t] = 'не найдена' + (': ' + '; '.join(why) if why else '')
    return out


ADULT_MODEL = {('conj', 'sclera'): 'eyes_defy', ('conj',): 'eyes_defy_conj', ('sclera',): 'eyes_defy_sclera'}
TISSUE_RU = {'conj': 'конъюнктива', 'sclera': 'склера'}


# ------------------------------------------------------------------ главный вход
def analyze_photo(photo, sex=None, age=None, pregnant=False, bundle=None, landmarker='auto'):
    """Фото глаза (оттянутое нижнее веко; можно уже вырезанную конъюнктиву) → dict:
        status   — 'ok' | 'retake' (снимок непригоден) | 'unreliable' (снимок не похож на обучающие / нет модели);
        stage    — где остановились: 'frame' (кадр технически непригоден), 'no_eye' (глаз не найден), 'no_tissue'
                   (глаз есть, тканей нет), 'quality' (ткани найдены, качество недостаточно), 'no_model' (для
                   этого набора тканей нет обученной модели), 'mismatch' (вырез vs снимок глаза), 'ood', 'ok';
        eye      — как найден глаз; tissue_status — {'conj': …, 'sclera': …}; tissues — какие ткани использованы;
        risk     — 'низкий' / 'средний' / 'высокий' или None; p_anemia — P(анемия) модели или None;
        hb_range — (от, до) г/л, экспериментально, или None; hb_note — почему цифры Hb нет / что она значит;
        model, kind — чем и по какому снимку оценено; reasons — почему такой ответ; issues — проблемы снимка;
        crop, conj, sclera — вырезка и маски (для показа, anemia.predict.overlay).
    Модель взрослых выбирается по пригодным тканям (обе / только конъюнктива / только склера): каждая обучена
    и проверена на своих признаках, недостающая ткань ничем не подменяется.
    landmarker — 'auto' (MediaPipe, если установлен), готовый детектор или None (без поиска лица)."""
    bundle = bundle if bundle is not None else load_bundle()
    sex = _sex(sex)
    key = pick_model(bundle, age)
    B = bundle['models'][key]
    r = {'status': 'retake', 'stage': 'frame', 'risk': None, 'p_anemia': None, 'hb_range': None, 'hb_note': None,
         'model': B['title'], 'features': None, 'eye': None, 'tissue_status': {},
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
    lm = None
    if landmarker == 'auto':
        lm, lm_note = face_landmarker()
        if lm is None:
            r['notes'].append(lm_note)
    elif landmarker:
        lm = landmarker
    det = _detect(rgb, lm)                  # ориентация — как есть (EXIF учтён); повёрнутый кадр меняет ответ
    crop, conj, sclera = det['crop'], det['conj'], det['sclera']
    ts = tissue_status(crop, det)
    good = tuple(t for t in ('conj', 'sclera') if ts[t] == 'пригодна')
    r.update(crop=crop, conj=conj, sclera=sclera, kind=det['kind'], eye=det['eye'], tissue_status=ts,
             tissues=list(good))
    if not det['use']:
        r['stage'] = 'no_eye' if det['eye'] is None else 'no_tissue'
        r['issues'].append('глаз не найден — снимите глаз с оттянутым нижним веком так, чтобы он был в фокусе'
                           if det['eye'] is None else
                           f'глаз найден ({det["eye"]}), но ни конъюнктива, ни склера не найдены — оттяните нижнее '
                           'веко и снимите глаз крупнее')
        r['issues'] += [f'{TISSUE_RU[t]}: {v}' for t, v in ts.items()]
        return r
    found = [det[t] for t in ('conj', 'sclera') if t in det['use']]
    whole = quality(crop, found[0], found[-1] if len(found) > 1 else np.zeros_like(found[0]), det['src_side'])
    if not good or whole:                   # проверка всех найденных тканей вместе — как раньше, не слабее
        r['stage'] = 'quality'
        r['issues'] = whole + [f'{TISSUE_RU[t]}: {v}' for t, v in ts.items() if v != 'пригодна']
        return r
    if det['kind'] == 'вырезанная ткань':
        if key == 'eyes_defy':
            r['status'], r['stage'] = 'unreliable', 'mismatch'
            r['issues'].append('это уже вырезанная ткань, а модель взрослых обучена на снимках глаза целиком — '
                               'пришлите фото глаза с оттянутым нижним веком')
            return r
        crop = resize_max(rgb, V12.CROP_SIDE)            # как обучались вырезы CP-AnemiC: весь кадр — ткань
        conj, sclera, good = np.ones(crop.shape[:2], bool), None, ('conj',)
    elif key == 'cp_anemic':
        r['status'], r['stage'] = 'unreliable', 'mismatch'
        r['issues'].append('модель детей до 5 лет обучена только на вырезанной конъюнктиве (CP-AnemiC); на снимке '
                           'глаза целиком она не проверялась — пришлите вырез конъюнктивы или сдайте анализ крови')
        return r
    else:
        key = ADULT_MODEL[good]
        if key not in bundle['models']:
            r['status'], r['stage'] = 'unreliable', 'no_model'
            r['issues'].append(f'пригодна только ткань «{", ".join(TISSUE_RU[t] for t in good)}», а модели для неё '
                               'в файле нет — перезапустите ячейку Я1 (scripts/train_core.py)')
            return r
        B = bundle['models'][key]
        r['model'] = B['title']
    r['tissues'] = list(good)
    f = V12.eye_features(crop, conj if 'conj' in good else None,
                         sclera if 'sclera' in good and sclera is not None else None)
    f['male'] = 1.0 if sex == 'M' else 0.0 if sex == 'F' else 0.5
    x = np.array([f.get(k, np.nan) for k in B['feats']], float)
    r['features'] = dict(zip(B['feats'], x.tolist()))
    need = CORE.SPECS[key]['need']
    if not np.isfinite(x[[i for i, k in enumerate(B['feats']) if k.startswith(need)]]).all():
        r['stage'] = 'quality'
        r['issues'].append('мало чистой ткани для признаков цвета (блики, тени, ресницы)')
        return r
    p, lv, hb, ood = (v[0] for v in CORE.apply_core(B, x[None]))
    if ood > CORE.OOD_MAX:
        r['status'], r['stage'] = 'unreliable', 'ood'
        r['issues'].append(f'{ood * 100:.0f}% признаков цвета далеко вне диапазона обучающих снимков '
                           '(другой свет, камера или не та ткань) — цифрам не верим, переснимите при дневном свете')
        return r
    r['status'], r['stage'], r['p_anemia'] = 'ok', 'ok', float(p)
    r['risk'] = CORE.RISK_LEVELS[int(lv)]
    r['reasons'].append({'eyes_defy': 'оценено по цвету конъюнктивы и склеры; цвет века дополнительно нормирован '
                                      'по белку глаза',
                         'eyes_defy_conj': 'оценено только по цвету конъюнктивы (склера непригодна — без поправки '
                                           'на свет по белку глаза)',
                         'eyes_defy_sclera': 'оценено только по цвету склеры (конъюнктива непригодна); у этой '
                                             'модели своя, отдельно проверенная точность — см. evaluate',
                         'cp_anemic': 'оценено по цвету вырезанной конъюнктивы'}[key])
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
        lines.append(f"На снимке: {r['kind']}; глаз: {r['eye'] or 'не найден'}")
        lines += [f"  {TISSUE_RU[t]}: {v}" for t, v in r['tissue_status'].items()]
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
