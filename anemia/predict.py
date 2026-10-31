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
from . import segment_baseline as SB
from . import v12_model as V12
from .io import resize_max
from .labs import is_pregnant, who_limits
from . import regions as R
from .regions import detect_regions, quality

MODEL_FILE = 'hemavision_core.joblib'
MIN_SIDE = 24            # px по короткой стороне (вырезы ткани бывают 171×34 — так было в исключённом CP-AnemiC);
                         # мелкую ткань ловит quality()
CHILD_MIN_AGE = 5        # лет; моложе — модели нет (CP-AnemiC исключён 2026-10-19): отказ, модель взрослых не подставляем
CHILD_REFUSAL = ('для детей младше 5 лет оценка по фото не проводится: модели, проверенной на детях, нет '
                 '(нормы ВОЗ и вид конъюнктивы у них другие) — сдайте общий анализ крови (Hb)')
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
        rgb[alpha == 0] = 0                        # вырезанная ткань: прозрачный фон — чёрный
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

        from .model_store import check_compat
        b = joblib.load(path)
        why = check_compat(b.get('compat'))
        if why:
            raise RuntimeError(f'модель {path} не подходит к текущему коду ({why}): запустите ячейку Я1 '
                               '(python scripts/train_core.py)')
        _BUNDLES[path] = b
    return _BUNDLES[path]


def _sex(sex):
    if sex is None:
        return None
    s = str(sex).strip().upper()
    return 'M' if s[:1] in ('M', 'М') else 'F' if s[:1] in ('F', 'Ж', 'W') else None


def child_refused(age):
    """Возраст младше 5 лет → оценки по фото нет (True). Возраст не указан → модель взрослых, как раньше."""
    return age is not None and age < CHILD_MIN_AGE


def pick_model(bundle, age):
    """Модель взрослых (Eyes-defy) для возраста ≥ 5 лет или неуказанного; младше 5 лет → None (оценки нет:
    детской модели нет, модель взрослых детям не подставляется)."""
    if child_refused(age):
        return None
    key = 'eyes_defy'
    if key not in bundle['models']:
        raise KeyError(f'в модели нет части «{key}» (обучена на {sorted(bundle["models"])})')
    return key


# ------------------------------------------------------------------ где глаз на снимке
FACE_MODEL_FILE = 'face_landmarker.task'
FACE_MODEL_URL = ('https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/'
                  'face_landmarker.task')
IRIS_MIN_R = 0.012       # самая мелкая радужка, которую ищем (доля короткой стороны кадра 1024 px → ≥ 6 px)
IRIS_SCLERA_L = 0.6      # мелкая радужка: L* внутри ≤ 0,6 L* склеры (у глаза на фото проверки 0,37–0,40; без размеченных данных)
SCLERA_LID_MARGIN = 1.0  # склера не дальше 1 r в сторону за край вывернутого века (конъюнктивы)
SCLERA_KEEP_R = 1.8      # ближе 1,8 r к центру радужки склеру по этому правилу не режем
IRIS_SHIFT = 1.0         # уточнение мелкой радужки сдвигает центр не дальше 1 радиуса
SMALL_IRIS_TRIES = 3     # сколько тёмных пятен проверить как «мелкую радужку» (по убыванию контраста края)
ZOOM_MAX_AREA = 0.5     # рамка вокруг радужки — меньше половины кадра, иначе это уже крупный план
WIDE_BG_MIN = 0.01       # тёмный фон у края > 1% кадра — кадр шире крупного плана (у Eyes-defy ≤ 0,4%)
ZOOM_WIDE = 1.4         # вторая попытка приближения: рамка в 1,4 раза шире (радиус по Хафу бывает занижен)
CONJ_SPREAD_MAX = 5.0    # части «конъюнктивы» дальше друг от друга (в радиусах радужки) — это не одно веко
CONJ_SPREAD_MAX_SEEN = 3.7   # максимум на 211 снимках Eyes-defy с радужкой (2026-10-18), запас ×1,35
SCLERA_AREA_MAX = 1.4    # площадь «склеры» в площадях радужки (π r²); выше — кожа/фон вокруг ложной радужки
SCLERA_AREA_MAX_SEEN = 1.15  # максимум на 192 снимках Eyes-defy со склерой
CONJ_FRAC_NO_IRIS = 0.05 # без радужки «конъюнктива» должна занимать ≥ 5% кадра (крупный план века)
CONJ_FRAC_NO_IRIS_SEEN = 0.098  # минимум у 5 снимков Eyes-defy без радужки
CONJ_DY_MIN = -2.5       # центр «конъюнктивы» выше центра радужки больше чем на 2,5 r — не нижнее веко
CONJ_DY_MIN_SEEN = -2.0  # минимум на 211 снимках Eyes-defy (там «радужкой» бывает и не радужка)
IRIS_BG_MAX = 0.1        # доля круга «радужки» на тёмном фоне у края кадра (волосы, фон); больше — это не радужка
IRIS_BG_MAX_SEEN = 0.02  # максимум на 211 снимках Eyes-defy с радужкой (2026-10-18)
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
    return _trim_to_eye(r) if ir is not None else r


def _trim_to_eye(r):
    """Рамка глаза (не весь кадр, путь обучения не трогается): из масок убирается то, что анатомически не может
    быть тканью этого глаза.
    1) Части масок целиком выше центра радужки: нижнее веко и видимая склера доходят до линии центра радужки
       (уголки глаза на её уровне); целиком выше — верхнее веко, ресницы, кожа.
    2) Склера сбоку дальше вывернутого века: веко с конъюнктивой тянется под всей глазной щелью, поэтому склера
       не выходит за его края больше чем на SCLERA_LID_MARGIN r. Пиксели ближе SCLERA_KEEP_R r к центру радужки
       не трогаются: склера, прилегающая к радужке, остаётся всегда, даже если конъюнктива найдена не целиком.
    Правила — по форме и положению, без порогов цвета; у Eyes-defy (если бы применялись) правило 2 задело бы
    5 снимков из 192 и убрало бы не больше 29% склеры."""
    cx, cy, ri = r['iris']
    new = {}
    for t in ('conj', 'sclera'):
        m = r[t]
        if t not in r['use'] or not m.any():
            continue
        n, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
        up = [i for i in range(1, n) if st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] < cy]
        if up:
            new[t] = (m & ~np.isin(lab, up), 'убраны участки выше центра радужки (веко, кожа)')
    conj = new['conj'][0] if 'conj' in new else r['conj']
    scl, why_s = new.get('sclera', (r['sclera'], ''))
    if 'conj' in r['use'] and conj.any() and 'sclera' in r['use'] and scl.any():
        xs = np.nonzero(conj)[1]
        h, w = scl.shape
        yy, xx = np.ogrid[0:h, 0:w]
        side = (xx < xs.min() - SCLERA_LID_MARGIN * ri) | (xx > xs.max() + SCLERA_LID_MARGIN * ri)
        out = scl & side & (np.hypot(xx - cx, yy - cy) > SCLERA_KEEP_R * ri)
        if out.any():
            scl = scl & ~out
            why_s = '; '.join(x for x in (why_s, f'убрано {out.sum()} px сбоку за краем века (кожа у уголка глаза)') if x)
            new['sclera'] = (scl, why_s)
    if not new:
        return r
    maps = R.color_maps(r['crop'])
    for t, (m, note) in new.items():
        ok, why = (R.check_conj(maps, m, sclera=scl) if t == 'conj' else R.check_sclera(maps, m, conj))
        r[t] = m if ok else np.zeros_like(m)
        r['notes'] = list(r['notes']) + [f'{TISSUE_RU[t]}: {note}']
        if not ok:
            r['use'] = r['use'] - {t}
            r['notes'].append(why)
    r['score'] = (2.0 if 'conj' in r['use'] else 0) + (1.0 if 'sclera' in r['use'] else 0) + 0.1
    return r


def _iris_box(iris, H, W, k=1.0):
    """Рамка глаза вокруг радужки (ширина глаза ≈ 2,2 диаметра радужки; пропорции как у face_eye_crops);
    k — во сколько раз шире."""
    cx, cy, ri = iris
    e = 4.4 * ri * k
    return (max(0, cx - 0.9 * e), max(0, cy - 0.6 * e), min(W, cx + 0.9 * e), min(H, cy + 1.0 * e))


def _eye_confirmed(z):
    """Глаз подтверждён целиком: обе ткани, они складываются в один глаз (_inconsistent), радужка намного темнее
    склеры (_dark_iris) и вывернутое веко лежит под радужкой (_lid_under_iris)."""
    return (z['iris'] is not None and z['use'] == {'conj', 'sclera'} and not _inconsistent(z) and _dark_iris(z)
            and _lid_under_iris(z))


def _zoom(rgb, scale, iris, H, W, how):
    """Разобрать рамку вокруг радужки. Тёмный круг в части кадра может быть чем угодно (ноздря, щель между
    пальцами, край волос), поэтому рамка принимается, только если глаз в ней подтверждён целиком (_eye_confirmed).
    Две попытки: рамка по уточнённой радужке; рамка в ZOOM_WIDE раз шире, радужка ищется заново внутри неё, как на
    крупном плане (радиус по Хафу бывает занижен, и тогда рамка режет веко).
    -> (результат или None, радужка рамки (cx, cy, r) в координатах кадра или None). Радужка возвращается и без
    результата, если в рамке нашлась тёмная радужка со склерой, но глаз не подтверждён (веко не под ней)."""
    box = _iris_box(iris, H, W)
    if box[2] - box[0] < 20 or box[3] - box[1] < 20 or (box[2] - box[0]) * (box[3] - box[1]) > ZOOM_MAX_AREA * H * W:
        return None, None
    ref = _refine_iris(rgb, scale, box, iris)
    seen = None
    for b, ir in ((_iris_box(ref, H, W), ref), (_iris_box(iris, H, W, ZOOM_WIDE), None)):
        z = _analyze_box(rgb, scale, b, ir, 'глаз в части кадра')
        if z['iris'] is None:
            continue
        if ir is None:
            z = _trim_to_eye(z)
            x0, x1, y0 = (int(round(v * scale)) for v in (b[0], b[2], b[1]))
            s = z['crop'].shape[1] / (x1 - x0)
            ir = ((z['iris'][0] / s + x0) / scale, (z['iris'][1] / s + y0) / scale, z['iris'][2] / s / scale)
        if _eye_confirmed(z):
            z['eye'] = how
            return z, ir
        if seen is None and z['sclera'].any() and _dark_iris(z):
            seen = ir
    return None, seen


def _refine_iris(rgb, scale, box, iris):
    """Уточнить мелкую радужку по рамке в полном разрешении: в уменьшенном кадре её центр бывает сдвинут на
    радиус (тогда зона поиска склеры съезжает на кожу у уголка глаза). Локальный перебор центра (до ±IRIS_SHIFT r)
    и радиуса (×0,75–1,25) по тому же контрасту края, что у find_iris. -> (cx, cy, r) в координатах кадра."""
    x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
    crop = R._prepare(rgb[y0:y1, x0:x1])
    s = crop.shape[1] / (x1 - x0)
    maps = R.color_maps(crop)
    L = cv2.GaussianBlur(maps['L'], (0, 0), 1.5)
    cx0, cy0, r0 = (iris[0] * scale - x0) * s, (iris[1] * scale - y0) * s, iris[2] * scale * s
    best = (R._iris_score(L, maps['C'], cx0, cy0, r0), cx0, cy0, r0)
    for step in (0.2, 0.1, 0.05):
        _, bx, by, br = best
        for dx in np.arange(-3, 4) * step * r0:
            for dy in np.arange(-3, 4) * step * r0:
                for k in (0.9, 1.0, 1.1):
                    x, y, r = bx + dx, by + dy, float(np.clip(br * k, 0.75 * r0, 1.25 * r0))
                    if np.hypot(x - cx0, y - cy0) > IRIS_SHIFT * r0:
                        continue
                    sc = R._iris_score(L, maps['C'], x, y, r)
                    if sc > best[0]:
                        best = (sc, x, y, r)
    _, x, y, r = best
    return ((x / s + x0) / scale, (y / s + y0) / scale, r / s / scale)


def _lid_under_iris(z):
    """Вывернутое нижнее веко (конъюнктива) лежит под радужкой: по горизонтали проходит под её центром,
    а его середина ниже центра радужки. Губы, щель между пальцами рядом с ложной «радужкой» так не лежат."""
    cx, cy, _ = z['iris']
    ys, xs = np.nonzero(z['conj'])
    return bool(len(xs)) and xs.min() <= cx <= xs.max() and ys.mean() > cy


def _dark_iris(z):
    """Радужка — отдельное тёмное пятно, а не край тёмной области:
    1) L* внутри круга ≤ IRIS_SCLERA_L × L* склеры (у ноздри или тени рядом «склера» — кожа, разница меньше);
    2) сразу под радужкой (1,15–1,6 r, в пределах её ширины) — склера или веко: светлее середины между радужкой
       и склерой. Прядь волос, край одежды продолжаются вниз тем же тёмным."""
    L = R.color_maps(z['crop'])['L']
    h, w = L.shape
    cx, cy, r = z['iris']
    yy, xx = np.ogrid[0:h, 0:w]
    inside = np.hypot(xx - cx, yy - cy) < 0.9 * r
    below = (np.abs(xx - cx) < 0.5 * r) & (yy > cy + 1.15 * r) & (yy < cy + 1.6 * r)
    if not inside.any() or not below.any():
        return False
    li, ls = np.median(L[inside]), np.median(L[z['sclera']])
    return li <= IRIS_SCLERA_L * ls and np.median(L[below]) >= (li + ls) / 2


def conj_spread(conj, r):
    """Разброс частей маски конъюнктивы: расстояние между крайними центрами компонент в радиусах радужки."""
    c = cv2.connectedComponentsWithStats(conj.astype(np.uint8), 8)[3][1:]
    return float(np.hypot(*(c.max(0) - c.min(0))) / r) if len(c) > 1 else 0.0


def _inconsistent(d):
    """Причина, по которой найденные ткани не складываются в один глаз, или ''. Правила:
    1) только «склера», без радужки и без конъюнктивы (не вырезка): привязать её к глазу нечем — светлым бывает
       лоб, щека, стена (у Eyes-defy такого нет ни разу: без радужки всегда есть конъюнктива);
       и то же для «конъюнктивы» меньше CONJ_FRAC_NO_IRIS кадра: без радужки принимаем только крупный план века;
    2) «склера» больше SCLERA_AREA_MAX площадей радужки (кожа или фон вокруг ложной «радужки»), если при этом
       радужка не подтверждена как глаз (_dark_iris и веко под ней): широко открытый глаз может дать и больше;
    3) центр «конъюнктивы» выше центра радужки больше чем на -CONJ_DY_MIN радиуса (это не нижнее веко);
    4) части «конъюнктивы» разнесены дальше CONJ_SPREAD_MAX радиусов радужки (веко и губы, веко и ноздря).
    5) круг «радужки» больше чем на IRIS_BG_MAX лежит на тёмном фоне у края кадра (волосы, фон за головой):
       радужка — часть глаза, фон в неё не заходит; такой круг — граница волос и кожи, а не глаз.
    Пороги — с запасом за пределами всех 216 снимков Eyes-defy (путь обучения ими не задевается)."""
    if d.get('iris') is None:
        if d.get('kind') == 'вырезанная ткань':
            return ''
        if d['use'] == {'sclera'}:
            return ('найдена только светлая область без радужки и без конъюнктивы: нельзя проверить, что это склера '
                    'глаза, а не кожа или фон')
        if 'conj' in d['use'] and d['conj'].mean() < CONJ_FRAC_NO_IRIS:
            return (f'радужка не найдена, а «конъюнктива» занимает {d["conj"].mean():.0%} кадра: это не крупный план '
                    f'века (у таких снимков Eyes-defy ≥ {CONJ_FRAC_NO_IRIS_SEEN:.0%}), нельзя проверить, что это веко')
        return ''
    cx, cy, r = d['iris']
    if d.get('crop') is not None:
        h, w = d['crop'].shape[:2]
        yy, xx = np.ogrid[0:h, 0:w]
        f = R.dark_border(d['crop'])[np.hypot(xx - cx, yy - cy) < r].mean()
        if f > IRIS_BG_MAX:
            return (f'«радужка» на {f:.0%} лежит на тёмном фоне у края кадра (волосы, фон; у снимков глаза не больше '
                    f'{IRIS_BG_MAX_SEEN:.0%}): это не радужка')
    if (d['sclera'] is not None and d['sclera'].sum() > SCLERA_AREA_MAX * np.pi * r * r
            and not (d.get('crop') is not None and _dark_iris(d) and d['conj'] is not None and _lid_under_iris(d))):
        return (f'«склера» в {d["sclera"].sum() / (np.pi * r * r):.1f} раза больше радужки (у снимков глаза не больше '
                f'{SCLERA_AREA_MAX_SEEN}): это кожа или фон, а найденная «радужка» — не радужка')
    if d['conj'] is None or not d['conj'].any():
        return ''
    dy = (np.nonzero(d['conj'])[0].mean() - cy) / r
    if dy < CONJ_DY_MIN:
        return (f'«конъюнктива» выше радужки ({-dy:.1f} её радиуса; у снимков глаза не выше {-CONJ_DY_MIN_SEEN}): '
                f'это не нижнее веко')
    k = conj_spread(d['conj'], r)
    if k > CONJ_SPREAD_MAX:
        return (f'«конъюнктива» — несколько областей далеко друг от друга ({k:.1f} радиуса радужки, у снимков глаза '
                f'не больше {CONJ_SPREAD_MAX_SEEN}): это не одно веко, ткани не связаны с глазом надёжно')
    return ''


def _rejected(d, why):
    """Тот же результат, но без тканей: лучше отказ, чем маска не на глазу."""
    d = dict(d, conj=np.zeros_like(d['conj']), sclera=np.zeros_like(d['sclera']), use=set(), score=0.0)
    d['notes'] = list(d['notes']) + [why]
    return d


def _detect(rgb, landmarker=None):
    """Глаз по шагам → результат regions (crop, conj, sclera, use, notes, kind, src_side) + eye — как найден глаз
    (None — не найден).
    1) Весь кадр — ровно как при обучении (все снимки Eyes-defy проходят этим путём, признаки совпадают).
       Обе ткани нашлись, они складываются в один глаз (_inconsistent) или это вырезанная ткань → готово.
       Не складываются (например, «конъюнктива» на веке и на губах) → весь кадр и его «радужку» не используем.
       В кадре тёмный фон у края (волосы, фон; у Eyes-defy его нет) → кадр шире крупного плана: радужка шага 1
       считается надёжной, только если глаз подтверждён (радужка тёмная, веко под ней).
    2) Иначе MediaPipe ищет лицо; нашёл и глаз не во весь кадр → рамки глаз по точкам лица. Если в кадре была
       надёжная крупная радужка, рамки берутся, только если в них нашлась склера и пригодных тканей больше (на
       крупных планах MediaPipe иногда «видит» лицо, которого нет).
    3) Надёжная крупная радужка была → результат шага 1 (радужку, найденную во всём кадре, для приближения не
       используем: на крупных планах ею бывает зрачок).
    4) Иначе ищем радужку не крупнее рамки ZOOM_MAX_AREA с надёжной (вдвое контрастнее обычного) границей и
       разбираем рамку вокруг неё; берём, только если глаз в рамке подтверждён целиком (_zoom).
    5) Кадр шире крупного плана, рамка нашла тёмную радужку со склерой, но веко не под ней, и «конъюнктива» шага 1
       тоже не под этой радужкой (снимок повёрнут, веко не оттянуто, ткани шага 1 на пальцах) → отказ с этой
       причиной. Иначе — результат шага 1 как есть.
    Любой результат с тканями, которые не складываются в один глаз, заменяется отказом (_rejected)."""
    img = resize_max(rgb, V12.DETECT_SIDE)
    H, W = img.shape[:2]
    scale = rgb.shape[1] / W
    cv2.setRNGSeed(0)                       # GrabCut случаен: тот же сид, что при обучении (evaluate._v12_compute)
    det = detect_regions(img, 'both')
    det['eye'] = 'глаз крупным планом' if det['iris'] is not None else 'весь кадр' if det['use'] else None
    bad = _inconsistent(det)
    if bad:
        det = _rejected(det, bad)
        det['eye'] = None
    big_iris = det['iris'] is not None and not bad
    # Eyes-defy — крупные планы глаза: тёмного фона у края кадра (волосы, фон за головой) там нет (≤ 0,4% кадра).
    # Есть такой фон — кадр шире крупного плана, и радужку во всём кадре find_iris находит ненадёжно (граница
    # волос, пальцы): результату шага 1 тогда верим, только если глаз подтверждён (радужка тёмная, веко под ней).
    wide = R.dark_border(img).mean() > WIDE_BG_MIN
    sure = not wide or big_iris and det['sclera'].any() and _dark_iris(det) and _lid_under_iris(det)
    if not bad and (det['kind'] == 'вырезанная ткань' or det['use'] == {'conj', 'sclera'} and sure):
        return det
    big_iris = big_iris and sure
    faces = R.face_eye_crops(img, landmarker) if landmarker else []
    if faces and max(c['box'][2] - c['box'][0] for c in faces) <= 0.5 * W:
        best = None
        for c in faces:
            r = _analyze_box(rgb, scale, c['box'], c['iris'], c['how'])
            why = _inconsistent(r)
            r = _rejected(r, why) if why else r
            if best is None or r['score'] > best['score']:
                best = r
        best['eye'] = 'лицо в кадре, глаз по точкам MediaPipe'
        if not big_iris or ('sclera' in best['use'] and len(best['use']) > len(det['use'])):
            return best                      # лицо целиком: остальной кадр не разбираем (губы, нос — не ткани)
    if big_iris:
        return det                           # крупная радужка есть: снимок как обучающие, ничего не угадываем
    seen = None
    for iris, con in _small_irises(img)[:SMALL_IRIS_TRIES]:
        z, ir = _zoom(rgb, scale, iris, H, W, f'мелкая радужка r={iris[2] * scale:.0f} px, контраст края {con:.0f}')
        if z:
            return z
        seen = seen or ir
    k = det['crop'].shape[1] / W            # кадр DETECT_SIDE → вырезка шага 1
    if (wide and seen and det['use']
            and not _lid_under_iris({'iris': tuple(v * k for v in seen), 'conj': det['conj']})):
        det = _rejected(det, 'глаз найден, но вывернутое нижнее веко не под радужкой: снимок повёрнут или веко не '
                             'оттянуто — держите телефон прямо и оттяните нижнее веко вниз')
        det['eye'] = 'радужка в части кадра, веко не под ней'
    return det


def _seg_steps(seg, crop):
    """Только для диагностики отказа: пиксели маски сегментатора после каждого шага SB.conj_mask (на кадре SEG_SIZE):
    raw — порог 0,5 и морфология; без нижнего края; крупные части; минимальная площадь, ниже которой маска пуста."""
    small = resize_max(np.ascontiguousarray(crop, np.float32), SB.SEG_SIZE)
    m = SB._raw_mask(seg, SB.pixel_features(small), 0.5)
    out = {'кадр': f'{small.shape[1]}x{small.shape[0]}', 'raw': int(m.sum())}
    _, lab = cv2.connectedComponents(m.astype(np.uint8), connectivity=8)
    bottom = np.unique(lab[-1])
    m &= ~np.isin(lab, bottom[bottom > 0])
    out['без нижнего края'] = int(m.sum())
    out['крупные части'] = int(SB._largest_parts(m).sum())
    out['мин. площадь'] = int(0.004 * m.size)
    return out


def learned_conj(det, seg, force=False):
    """Конъюнктива ядра — сегментатор по маскам врачей Eyes-defy (anemia.segment_baseline) на вырезке, где _detect
    нашёл глаз; та же функция даёт маски признаков обучения (scripts/evaluate.py core_features), поэтому признаки
    снимка совпадают с обучающими. Пороговая «конъюнктива» regions нужна только для поиска глаза: она захватывает
    склеру, кожу и пальцы (IoU с врачами ≈ 0,30, разовый замер). Сегментатор конъюнктиву не нашёл — её нет; затем ткани ещё раз
    проверяются на согласие с радужкой (_inconsistent). Конъюнктивы нет — отказ целиком: «склера» без вывернутого
    века не привязана к глазу надёжно (на фото проверки ею оказывалась кожа щёк при ложной «радужке»)."""
    if (not det['use'] and not force) or det['kind'] == 'вырезанная ткань':
        return det
    conj, steps = SB.conj_mask(seg, det['crop']), _seg_steps(seg, det['crop'])
    d = dict(det, conj=conj, use=(set(det['use']) - {'conj'}) | ({'conj'} if conj.any() else set()),
             notes=[n for n in det['notes'] if n != 'оценка по одной ткани'], seg_steps=steps)
    if not conj.any():                      # без века оценки нет: «склера» без него бывает кожей (у Eyes-defy
        return _rejected(d, 'конъюнктива не найдена сегментатором (обучен на масках врачей): нижнее веко не '
                            'вывернуто или не в кадре — оттяните его вниз')   # веко находится на всех снимках)
    why = _inconsistent(d)
    return _rejected(d, why) if why else d


ZOOM_CONJ_BOXES = (   # рамки повторного поиска века: (название, доли «радиуса» найденного круга влево/вправо, вверх, вниз)
    ('по контуру глаза', 1.2, 0.8, 1.0),   # круг шага 1 бывает всем контуром глаза (r в 3 раза больше радужки)
    ('по радужке', None, None, None),      # круг — радужка: рамка глаза _iris_box
)


def _zoom_conj(rgb, det, seg, H, W):
    """Повторный поиск века на рамке вокруг глаза, найденного шагом 1 во весь кадр. Сегментатор века обучен на
    крупных планах Eyes-defy (глаз во весь кадр); его признаки пикселей зависят от кадра (баланс «серый мир» и
    яркость по всему кадру, положение пикселя), и когда глаз занимает треть кадра, а остальное — кожа и пальцы,
    веко он не находит. Рамка вырезается из исходного фото (_analyze_box), обе ткани ищутся в ней заново, так что
    маски и вырезка в одних координатах; проверки те же (learned_conj → _inconsistent), пороги не меняются.
    Склера здесь не используется (оценка по конъюнктиве): радиус радужки на такой рамке неизвестен, и пороговая
    «склера» на фото проверки ложилась на кожу у края рамки.
    → (результат с веком или None, строки диагностики)."""
    k = W / det['crop'].shape[1]                     # вырезка шага 1 → кадр DETECT_SIDE
    cx, cy, r = (v * k for v in det['iris'])
    scale = rgb.shape[1] / W
    log = []
    for name, a, up, down in ZOOM_CONJ_BOXES:
        box = _iris_box((cx, cy, r), H, W) if a is None else (cx - a * r, cy - up * r, cx + a * r, cy + down * r)
        box = (max(0, box[0]), max(0, box[1]), min(W, box[2]), min(H, box[3]))
        x0, y0, x1, y1 = (int(round(v * scale)) for v in box)
        if x1 - x0 < 40 or y1 - y0 < 40:
            log.append(f'рамка {name}: меньше 40 px — пропуск')
            continue
        z = learned_conj(_analyze_box(rgb, scale, box, (cx, cy, r), 'глаз в части кадра'), seg, force=True)
        ok = 'conj' in z['use']
        if ok and 'sclera' in z['use']:   # масштаб глаза неизвестен (круг — контур глаза или радужка): проверить
            z = dict(z, sclera=np.zeros_like(z['sclera']), use=z['use'] - {'sclera'},   # «склеру» нечем, на
                     notes=z['notes'] + ['склера на рамке повторного поиска не используется: её нечем проверить'])
        log.append(f"рамка {name}: x {x0}–{x1}, y {y0}–{y1} px фото; сегментатор {z.get('seg_steps')}; "
                   + ('веко принято' if ok else 'отказ: ' + (z['notes'][-1] if z['notes'] else 'тканей нет')))
        if ok:
            z['eye'] = f"{det['eye']}; веко найдено на рамке {name}"
            return z, log
    return None, log


def conj_source(det0, det):
    """Строка диагностики: какая маска конъюнктивы идёт в признаки. В расчёт идёт только маска сегментатора века
    (learned_conj, на вырезке det['crop']); пороговая маска regions (det0) нужна лишь для поиска глаза. Числа
    seg_steps — на кадре сегментатора SEG_SIZE, а не на вырезке, поэтому с пикселями вырезки их не сравнивать."""
    m = det.get('conj')
    used = int(m.sum()) if m is not None and 'conj' in det.get('use', ()) else 0
    h, w = det['crop'].shape[:2]
    h0, w0 = det0['crop'].shape[:2]
    return (f'маска конъюнктивы в расчёте: сегментатор века — {used} px на вырезке {w}x{h} '
            f'({100 * used / (h * w):.1f}% кадра){"" if used else " (нет — оценки по конъюнктиве не будет)"}; '
            f'пороговая маска поиска глаза — {int(det0["conj"].sum())} px на вырезке {w0}x{h0}, в расчёт НЕ идёт')


def _small_irises(img):
    """Кандидаты в мелкую радужку [((cx, cy, r), контраст края)], лучшие первыми. Окружности Хафа всех трёх
    порогов (find_iris останавливается на первом пороге, где нашлись хоть какие-то окружности, и при сдвиге или
    отражении кадра радужку пропускает) + ответ find_iris. Радужка с надёжным краем (IRIS_CONTRAST) и целиком в
    кадре с запасом 1,5 r: мелкий глаз помещается в кадр, а у края тёмными бывают волосы, тень, угол.
    Радиус — до самого крупного, при котором рамка глаза (_iris_box: 7,92 r × 7,04 r) не больше ZOOM_MAX_AREA
    кадра: глаз средней величины (радужка ~0,08 короткой стороны) тоже приближается, если шаг 1 его не нашёл."""
    H, W = img.shape[:2]
    side = min(H, W)
    r_max = np.sqrt(ZOOM_MAX_AREA * H * W / (7.92 * 7.04))
    maps = R.color_maps(img)
    L = cv2.GaussianBlur(maps['L'], (0, 0), 1.5)
    u8 = np.clip(L * 2.55, 0, 255).astype(np.uint8)
    cands = []
    lo = IRIS_MIN_R * side
    while lo < r_max:                       # радиусы полосами ×2: подавление соседей в Хафе общее для всех радиусов,
        hi = min(2 * lo, r_max)             # и крупные круги (пальцы, контур глаза) иначе прячут радужку рядом
        for p2 in (30, 22, 16):
            c = cv2.HoughCircles(u8, cv2.HOUGH_GRADIENT, dp=1.5, minDist=side / 6, param1=60, param2=p2,
                                 minRadius=int(lo), maxRadius=int(hi))
            cands += [] if c is None else [tuple(float(v) for v in x) for x in c[0][:8]]
        lo = hi
    best = R.find_iris(img, maps, min_r=IRIS_MIN_R, max_r=r_max / side, min_contrast=IRIS_CONTRAST)
    cands += [best] if best else []
    out = []
    for cx, cy, r in cands:
        if min(cx, cy, W - cx, H - cy) <= 1.5 * r or r > r_max:
            continue
        if any(np.hypot(cx - o[0][0], cy - o[0][1]) < 0.5 * max(r, o[0][2]) for o in out):
            continue                                    # та же окружность с другого порога
        con = R._iris_score(L, maps['C'], cx, cy, r)
        if con > IRIS_CONTRAST:
            central = 1 - min(1, np.hypot((cx - W / 2) / W, (cy - H / 2) / H) * 1.2)
            out.append(((cx, cy, r), con, con * (0.5 + central)))
    out.sort(key=lambda o: -o[2])
    return [(i, c) for i, c, _ in out]


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


def tissue_route(det):
    """Снимок глаза → (модель взрослых, которой его оценит analyze_photo, или None; причина отказа). Те же шаги,
    что в analyze_photo: ткани не найдены → отказ; проверка качества каждой ткани и всех вместе (tissue_status,
    quality) → отказ или выбор модели по пригодным тканям. evaluate --task core и train_core применяют её к
    Eyes-defy, чтобы обучение и метрики шли на тех же снимках, которые принимает анализ фото."""
    if not det['use']:
        return None, 'нет тканей: ' + (det['notes'][-1] if det['notes'] else 'не найдены')
    ts = tissue_status(det['crop'], det)
    good = tuple(t for t in ('conj', 'sclera') if ts[t] == 'пригодна')
    found = [det[t] for t in ('conj', 'sclera') if t in det['use']]
    whole = quality(det['crop'], found[0], found[-1] if len(found) > 1 else np.zeros_like(found[0]), det['src_side'])
    if not good or whole:
        return None, 'качество: ' + '; '.join(whole + [f'{TISSUE_RU[t]}: {v}' for t, v in ts.items() if v != 'пригодна'])
    return ADULT_MODEL[good], ''
TISSUE_RU = {'conj': 'конъюнктива', 'sclera': 'склера'}


# ------------------------------------------------------------------ главный вход
def analyze_photo(photo, sex=None, age=None, pregnant=None, bundle=None, landmarker='auto'):
    """Фото глаза (оттянутое нижнее веко) → dict. Уже вырезанная ткань не оценивается (stage 'mismatch'),
    ребёнок младше 5 лет — тоже (stage 'age', до загрузки модели и разбора фото). pregnant: True / False / None.
        status   — 'ok' | 'retake' (снимок непригоден) | 'unreliable' (снимок не похож на обучающие / нет модели);
        stage    — где остановились: 'frame' (кадр технически непригоден), 'no_eye' (глаз не найден), 'no_tissue'
                   (глаз есть, тканей нет), 'quality' (ткани найдены, качество недостаточно), 'no_model' (для
                   этого набора тканей нет обученной модели), 'mismatch' (вырез вместо снимка глаза), 'age' (младше 5 лет), 'ood', 'ok';
        eye      — как найден глаз; tissue_status — {'conj': …, 'sclera': …}; tissues — какие ткани использованы;
        risk     — 'низкий' / 'средний' / 'высокий' или None; p_anemia — P(анемия) модели или None;
        hb_range — (от, до) г/л, экспериментально, или None; hb_note — почему цифры Hb нет / что она значит;
        model, kind — чем и по какому снимку оценено; reasons — почему такой ответ; issues — проблемы снимка;
        crop, conj, sclera — вырезка и маски (для показа, anemia.predict.overlay).
    Модель взрослых выбирается по пригодным тканям (обе / только конъюнктива / только склера): каждая обучена
    и проверена на своих признаках, недостающая ткань ничем не подменяется.
    landmarker — 'auto' (MediaPipe, если установлен), готовый детектор или None (без поиска лица)."""
    sex = _sex(sex)
    r = {'status': 'retake', 'stage': 'frame', 'risk': None, 'p_anemia': None, 'hb_range': None, 'hb_note': None,
         'model': None, 'features': None, 'eye': None, 'tissue_status': {},
         'tissues': [], 'kind': None, 'reasons': [], 'issues': [], 'notes': [], 'crop': None, 'conj': None,
         'sclera': None, 'disclaimer': DISCLAIMER, 'diag': []}
    if child_refused(age):
        r['status'], r['stage'], r['model'] = 'unreliable', 'age', 'нет (младше 5 лет)'
        r['issues'].append(CHILD_REFUSAL)
        return r
    bundle = bundle if bundle is not None else load_bundle()
    key = pick_model(bundle, age)
    B = bundle['models'][key]
    r['model'] = B['title']
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
    if bundle.get('conj_seg') is None:
        r['status'], r['stage'] = 'unreliable', 'no_model'
        r['issues'].append('в файле модели нет сегментатора конъюнктивы (модель старше кода) — перезапустите ячейку Я1 '
                           '(python scripts/train_core.py)')
        return r
    img_hw = resize_max(rgb, V12.DETECT_SIDE).shape[:2]
    det0 = _detect(rgb, lm)                 # ориентация — как есть (EXIF учтён)
    det = learned_conj(det0, bundle['conj_seg'])
    diag = [f'фото {rgb.shape[1]}x{rgb.shape[0]} px; поиск глаза: {det0["eye"] or "не найден"}; вырезка '
            f'{det0["crop"].shape[1]}x{det0["crop"].shape[0]} px (сторона источника {det0["src_side"]} px); радужка '
            + (str(tuple(round(v) for v in det0['iris'])) if det0.get('iris') is not None else 'нет'),
            f'поиск тканей (пороги): {sorted(det0["use"]) or "ничего"}; конъюнктива {int(det0["conj"].sum())} px, '
            f'склера {int(det0["sclera"].sum())} px; заметки: {det0["notes"] or "-"}',
            f'сегментатор века, шаги на его кадре {SB.SEG_SIZE} px (не пиксели вырезки): ' + (str(det['seg_steps']) if 'seg_steps' in det else
                                               'не запускался (тканей нет или вырезка ткани)')]
    if ('conj' not in det['use'] and det0['kind'] != 'вырезанная ткань' and det0['eye'] is not None
            and det0.get('iris') is not None and det0['src_side'] == max(img_hw)):
        z, log = _zoom_conj(rgb, det0, bundle['conj_seg'], *img_hw)
        diag += log
        det = z or det
    crop, conj, sclera = det['crop'], det['conj'], det['sclera']
    diag.append(conj_source(det0, det))
    r['diag'] = diag
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
    r['diag'].append(f'QC: {ts}; общая проверка качества: {whole or "пройдена"}')
    if not good or whole:                   # проверка всех найденных тканей вместе — как раньше, не слабее
        r['stage'] = 'quality'
        r['issues'] = whole + [f'{TISSUE_RU[t]}: {v}' for t, v in ts.items() if v != 'пригодна']
        return r
    if det['kind'] == 'вырезанная ткань':
        r['status'], r['stage'] = 'unreliable', 'mismatch'
        r['issues'].append('это уже вырезанная ткань, а модель взрослых обучена на снимках глаза целиком — '
                           'пришлите фото глаза с оттянутым нижним веком')
        return r
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
    lv_p = int(lv)
    if B['hb_ok']:                          # согласование с Hb: весь диапазон ниже нормы ВОЗ → не ниже «высокого»
        normal = who_limits(sex, age, pregnant).normal
        lv, up = (int(v[0]) for v in CORE.reconcile([lv], [hb], B['hb_halfwidth'], True, [normal]))
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
                                             'модели своя, отдельно проверенная точность — см. evaluate'}[key])
    cal = ('калибрована (Platt, вложенная CV на обучающих пациентах)' if B.get('cal') is not None else
           'НЕ калибрована: файл модели старше кода — перезапустите ячейку Я1')
    if 'cal' in B:
        low = (f'< {B["t_lo"]:.2f} — низкий (так «средний + высокий» находят ≥ {B["low_sens_target"]:.0%} больных)'
               if np.isfinite(B['t_lo']) else 'уровня «низкий» у этой модели нет')
    else:
        low = (f'≤ {B["t_lo"]:.2f} — низкий' if np.isfinite(B['t_lo']) else 'уровня «низкий» у этой модели нет')
    high = (f'≥ {B["t_hi"]:.2f} — высокий (≥ {B["zone_target"]:.0%} больных)' if np.isfinite(B['t_hi'])
            else 'уровня «высокий» у этой модели нет')
    r['reasons'].append(f'P(анемия) = {p:.2f}, {cal}; пороги модели: {low}, {high} (подобраны вложенной CV на '
                        'обучающих пациентах; на невиданных пациентах — см. evaluate)')
    if B['hb_ok']:
        hw = B['hb_halfwidth']
        lo, hi = CORE.hb_interval(hb, hw)
        r['hb_range'] = (float(lo), float(hi))
        if lv > lv_p:
            r['risk'] = CORE.RISK_LEVELS[lv]
            r['reasons'].append(f'по P уровень «{CORE.RISK_LEVELS[lv_p]}», но весь диапазон Hb {lo:.0f}–{hi:.0f} г/л ниже '
                                f'нормы ВОЗ {normal:.0f} г/л → уровень поднят до «высокий»')
        elif lv == 2 and lo >= normal:
            r['reasons'].append(f'модели расходятся: по P «высокий», а диапазон Hb {lo:.0f}–{hi:.0f} г/л не ниже нормы '
                                f'ВОЗ {normal:.0f} г/л; уровень не понижается (скрининг) — нужен анализ крови')
        r['hb_note'] = (f'экспериментально: на вложенной CV обучающих пациентов в 80% случаев реальный Hb был '
                        f'в пределах ±{hw:.0f} г/л от прогноза (MAE {B["hb_mae_cv"]:.0f} г/л против {B["hb_mae_const_cv"]:.0f} у константы)')
    else:
        r['hb_note'] = (f'цифра Hb не выдаётся: на проверке модель Hb почти не лучше константы '
                        f'(MAE {B["hb_mae_cv"]:.0f} против {B["hb_mae_const_cv"]:.0f} г/л)')
    if is_pregnant(pregnant) or (age is not None and CHILD_MIN_AGE <= age < 18):
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
    if r.get('diag'):
        lines += ['Диагностика:'] + [f'  {x}' for x in r['diag']] + [f"  итог: {r['status']} / этап {r['stage']}"]
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
