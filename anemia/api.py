"""Единая точка входа для приложения: фото глаза → результат скрининга риска анемии (не диагноз).

    from anemia.api import load_models, screen
    models = load_models('models/hemavision_core.joblib')        # один раз при старте приложения
    res = screen(jpeg_bytes, sex='F', age=30, pregnant=None, models=models)
    res.risk, res.p_anemia, res.hb_interval_g_l, res.to_json()

Вся работа — в существующем ядре anemia.predict.analyze_photo (поиск глаза, сегментация конъюнктивы и склеры,
нормировка цвета по склере, QC, OOD, модель и пороги из scripts/train_core.py). Здесь только проверка ввода,
загрузка модели и перевод ответа ядра в Result; новых порогов и правил нет.
Лабораторный Hb сюда не передаётся (отдельный путь — anemia.labs.lab_hb_report).
"""
import base64
import io
import json
import math
import numbers
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import predict as PR

RISK_UNKNOWN = 'не определён'
TISSUE = {('conj', 'sclera'): 'обе', ('conj',): 'конъюнктива', ('sclera',): 'склера'}
AGE_MAX = 120
SYMPTOMS_NOTE = 'симптомы приняты и записаны в результат, но на риск пока не влияют (логики учёта симптомов нет)'
# Проблемы качества — подстроки сообщений существующих проверок (predict.frame_issues, regions.quality)
BLUR = ('размыт',)
DARK = ('темно', 'почти чёрный')
BRIGHT = ('пересвет', 'пересвеч')


@dataclass(frozen=True)
class Models:
    """Загруженная модель ядра (bundle scripts/train_core.py) + детектор лица MediaPipe (или None)."""
    bundle: dict
    landmarker: object = None
    version: str = ''
    notes: tuple = ()


@dataclass
class Result:
    risk: str                          # 'низкий' | 'средний' | 'высокий' | 'не определён'
    p_anemia: float | None             # P(анемия) модели, 0–1; None, если риск не определён
    hb_interval_g_l: list | None       # [низ, верх] г/л, экспериментально; None, если модель Hb не лучше константы
    tissue: str | None                 # 'конъюнктива' | 'склера' | 'обе'; None, если риск не определён
    reason: str                        # почему выбран метод (или почему оценки нет)
    quality: dict                      # {'sharpness', 'light', 'tissue_visible': {'conjunctiva', 'sclera'}}
    warnings: list
    masks: dict                        # {'conjunctiva': bool H×W | None, 'sclera': …} в координатах crop
    crop: np.ndarray | None            # вырезка глаза uint8 RGB, на которой лежат маски
    model_version: str
    status: str = 'retake'             # 'ok' | 'retake' (переснять) | 'unreliable' (снимок не похож на обучающие)
    stage: str = 'frame'               # где остановилось ядро (см. predict.analyze_photo)
    hb_note: str | None = None
    symptoms: dict | None = None
    disclaimer: str = PR.DISCLAIMER
    diag: list = field(default_factory=list)

    def to_dict(self, images=True):
        """Словарь для JSON. images=True — маски и вырезка как PNG в base64, False — без них."""
        d = {k: getattr(self, k) for k in self.__dataclass_fields__ if k not in ('masks', 'crop')}
        if images:
            d['masks'] = {k: _png(v) for k, v in self.masks.items()}
            d['crop'] = _png(self.crop)
        return d

    def to_json(self, images=True, **kw):
        return json.dumps(self.to_dict(images), ensure_ascii=False, **kw)


def _png(a):
    if a is None:
        return None
    a = np.asarray(a)
    img = Image.fromarray(a.astype(np.uint8) * 255 if a.dtype == bool else a.astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode('ascii')


# ------------------------------------------------------------------ модель
def load_models(path, face_model=None):
    """Файл модели ядра → Models. Вызывать один раз; дальше передавать в screen(models=…).
    face_model — файл MediaPipe face_landmarker.task (по умолчанию рядом с моделью); не скачивается: нет файла
    или mediapipe — глаз ищется по всему кадру и по радужке (так же, как в ядре без MediaPipe)."""
    path = os.fspath(path)
    bundle = PR.load_bundle(path)               # проверка совместимости с кодом — там же
    if bundle.get('conj_seg') is None or 'eyes_defy' not in bundle.get('models', {}):
        raise RuntimeError(f'модель {path} без сегментатора конъюнктивы или модели взрослых: обучена старым кодом — '
                           'запустите ячейку Я1 (python scripts/train_core.py)')
    lm, note = PR.face_landmarker(face_model or os.path.join(os.path.dirname(os.path.abspath(path)),
                                                             PR.FACE_MODEL_FILE), download=False)
    return Models(bundle=bundle, landmarker=lm, version=model_version(bundle), notes=(note,))


def model_version(bundle):
    """'core-<sha1 кода признаков>+<коммит обучения>'."""
    code = (bundle.get('compat') or {}).get('code') or '?'
    return f"core-{code}+{str(bundle.get('commit') or '?')[:12]}"


# ------------------------------------------------------------------ проверка ввода
def _check_image(image):
    if image is None:
        raise TypeError('image: нужно фото (bytes, numpy.ndarray H×W×3 или PIL.Image), получено None')
    if isinstance(image, (str, os.PathLike)):
        raise TypeError('image: API не читает файлы — передайте содержимое (bytes) или массив')
    if isinstance(image, (bytearray, memoryview)):
        return bytes(image)
    if not isinstance(image, (bytes, np.ndarray, Image.Image)):
        raise TypeError(f'image: ожидаются bytes, numpy.ndarray или PIL.Image, получено {type(image).__name__}')
    return image


def _check_sex(sex):
    if sex is None:
        return None
    if not isinstance(sex, str):
        raise TypeError(f'sex: строка "M"/"F" (или "М"/"Ж") либо None, получено {type(sex).__name__}')
    s = PR._sex(sex)
    if s is None:
        raise ValueError(f'sex: непонятное значение {sex!r}; допустимо "M"/"F", "М"/"Ж" или None (не указан)')
    return s


def _check_age(age):
    if age is None:
        return None
    if isinstance(age, (bool, np.bool_)) or not isinstance(age, numbers.Real):
        raise TypeError(f'age: число лет (int/float) или None, получено {type(age).__name__} {age!r}')
    age = float(age)
    if not math.isfinite(age) or not 0 <= age <= AGE_MAX:
        raise ValueError(f'age: возраст в годах от 0 до {AGE_MAX}, получено {age!r}')
    return age


def _check_pregnant(pregnant, sex):
    if pregnant is None:
        return None
    if not isinstance(pregnant, (bool, np.bool_)):
        raise TypeError(f'pregnant: True / False / None (неизвестно), получено {type(pregnant).__name__} {pregnant!r}')
    if pregnant and sex == 'M':
        raise ValueError('pregnant=True при sex="M"')
    return bool(pregnant)


def _check_symptoms(symptoms):
    """None, список названий симптомов или {название: bool/число/None} → dict или None."""
    if symptoms is None:
        return None
    if isinstance(symptoms, dict):
        items = symptoms.items()
    elif isinstance(symptoms, (list, tuple, set, frozenset)):
        items = ((s, True) for s in symptoms)
    else:
        raise TypeError(f'symptoms: список строк или словарь {{симптом: значение}}, получено {type(symptoms).__name__}')
    out = {}
    for k, v in items:
        if not isinstance(k, str) or not k.strip():
            raise TypeError(f'symptoms: название симптома — непустая строка, получено {k!r}')
        if v is not None and not isinstance(v, (bool, np.bool_, numbers.Real)):
            raise TypeError(f'symptoms[{k!r}]: bool, число или None, получено {type(v).__name__}')
        out[k.strip()] = bool(v) if isinstance(v, (bool, np.bool_)) else None if v is None else float(v)
    return out


# ------------------------------------------------------------------ перевод ответа ядра
def _quality(r):
    """Качество — из сообщений существующих проверок ядра (новых измерений нет). None — не проверялось:
    ядро остановилось раньше (кадр, глаз, ткани не найдены, возраст)."""
    txt = ' | '.join(r['issues'] + list(r['tissue_status'].values())).lower()
    checked = r['stage'] not in ('frame', 'no_eye', 'no_tissue', 'age')
    sharp = 'размыто' if any(s in txt for s in BLUR) else 'ok' if checked else None
    light = ('темно' if any(s in txt for s in DARK) else 'пересвет' if any(s in txt for s in BRIGHT)
             else 'ok' if checked else None)
    visible = {name: (not v.startswith('не найдена')) if v else None
               for t, name in (('conj', 'conjunctiva'), ('sclera', 'sclera'))
               for v in [r['tissue_status'].get(t)]}
    return {'sharpness': sharp, 'light': light, 'tissue_visible': visible}


def _reason(r):
    if r['status'] != 'ok':
        return '; '.join(r['issues']) or PR.STATUS_RU[r['status']]
    ts = '; '.join(f'{PR.TISSUE_RU[t]}: {v}' for t, v in r['tissue_status'].items())
    return f'{ts} → {r["reasons"][0]}'


def to_result(r, version, symptoms=None):
    """dict из predict.analyze_photo → Result."""
    ok = r['status'] == 'ok'
    crop = None if r['crop'] is None else (np.clip(r['crop'], 0, 1) * 255).round().astype(np.uint8)
    masks = {name: None if r[t] is None else np.asarray(r[t], bool)
             for t, name in (('conj', 'conjunctiva'), ('sclera', 'sclera'))}
    warnings = list(r['notes']) if ok else list(r['issues']) + list(r['notes'])
    if symptoms is not None:
        warnings.append(SYMPTOMS_NOTE)
    return Result(risk=r['risk'] if ok else RISK_UNKNOWN,
                  p_anemia=float(r['p_anemia']) if ok else None,
                  hb_interval_g_l=list(r['hb_range']) if ok and r['hb_range'] else None,
                  tissue=TISSUE[tuple(r['tissues'])] if ok else None,
                  reason=_reason(r), quality=_quality(r), warnings=warnings, masks=masks, crop=crop,
                  model_version=version, status=r['status'], stage=r['stage'],
                  hb_note=r['hb_note'] if ok else None, symptoms=symptoms, diag=list(r.get('diag') or []))


# ------------------------------------------------------------------ главный вход
def screen(image, sex, age, pregnant, symptoms=None, *, models):
    """Фото глаза с оттянутым нижним веком → Result.
    image: bytes (JPEG/PNG), numpy.ndarray H×W×3 (uint8 или float 0–1) или PIL.Image; sex: 'M'/'F'/'М'/'Ж'/None;
    age: лет (число) или None; pregnant: True/False/None (неизвестно); symptoms: None, ['слабость', …] или
    {'слабость': True, …} — проверяются и возвращаются, на риск пока не влияют; models: из load_models().
    Неверный ввод → TypeError / ValueError. Непригодный снимок, не глаз, ребёнок младше 5 лет, нетипичный
    снимок → risk='не определён' и причина в reason, без исключения и без цифр."""
    if not isinstance(models, Models):
        raise TypeError('models: результат load_models(path)')
    image = _check_image(image)
    sex = _check_sex(sex)
    age = _check_age(age)
    pregnant = _check_pregnant(pregnant, sex)
    symptoms = _check_symptoms(symptoms)
    r = PR.analyze_photo(image, sex=sex, age=age, pregnant=pregnant, bundle=models.bundle,
                         landmarker=models.landmarker)
    return to_result(r, models.version, symptoms)
