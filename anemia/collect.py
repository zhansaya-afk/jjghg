"""Сбор своих фото со смартфона + лабораторный Hb того же дня → data/phone/ (формат anemia.data.load_phone).

Логика без интерфейса (интерфейс — app/collect.py). Правила:
- id участника только обезличенный: 'P' + 6 hex-символов, генерируется случайно; ФИО, телефон, имя исходного файла
  нигде не сохраняются; EXIF/GPS/модель телефона отбрасываются (пиксели пересохраняются в PNG без метаданных).
- Hb — из бланка анализа (labs.hb_to_g_l), хранится в hb.csv; оценка модели по фото здесь не считается и не пишется.
- Дети младше 5 лет — отказ, как в ядре (predict.child_refused): модели для них нет.
- Проверка снимка — те же функции, что в ядре: predict.frame_issues (кадр) и regions.quality (резкость, свет,
  размер ткани). Предложение маски конъюнктивы — regions.detect_regions (без обучения), дальше правка кистью.
"""
import csv
import os
import re
import secrets

import cv2
import numpy as np
from PIL import Image

from . import regions as R
from .io import resize_max, resize_to
from .labs import hb_to_g_l, norm_pregnant
from .predict import CHILD_MIN_AGE, frame_issues, load_photo

ID_RE = re.compile(r'^P[0-9a-f]{6}$')
EYES = ('L', 'R')
MAX_SAVE_SIDE = 2000          # px: сохраняемое фото (ядро работает на вырезке 320 px; полный 12 Мп кадр не нужен)
MAX_LAB_GAP_DAYS = 1          # анализ крови и фото — в один день (допуск ±1 сутки)
BLOCKING = ('снимок размыт', 'слишком темно', 'пересвет / вспышка', 'ткань слишком мелкая в кадре — снимите ближе')
HB_COLUMNS = ['file', 'patient_id', 'hb_g_l', 'sex', 'age', 'pregnant', 'mask_conj', 'eye', 'lab_gap_days',
              'blur', 'mask_source']

PROTOCOL = """**Протокол съёмки**
1. Анализ крови (Hb) — в тот же день, что и фото. Нужно письменное согласие участника.
2. Свет: дневной у окна, без прямого солнца; **без вспышки**, без фильтров и «улучшайзеров» камеры.
3. Оттяните нижнее веко пальцем, взгляд вверх: розовая слизистая должна быть видна полосой.
4. Телефон в 10–15 см от глаза, фокус — тапом по веку. Веко занимает заметную часть кадра.
5. По 2 снимка каждого глаза. Лицо целиком не нужно: только глаз крупно."""


class CollectError(ValueError):
    """Запись нельзя сохранить: понятное сообщение для оператора."""


def new_id(existing=()):
    """Случайный обезличенный id участника ('P3fa9c1'), не совпадающий с уже существующими."""
    existing = set(existing)
    while True:
        pid = 'P' + secrets.token_hex(3)
        if pid not in existing:
            return pid


def check_id(pid):
    pid = (pid or '').strip()
    if not ID_RE.match(pid):
        raise CollectError('id участника — только сгенерированный кнопкой (вида P3fa9c1); имена и номера не вводите')
    return pid


def parse_age(age):
    """Возраст в годах: число или строка («30», «4,5») → float; пусто → ошибка (для сбора возраст обязателен)."""
    if age is None or (isinstance(age, str) and not age.strip()):
        raise CollectError('укажите возраст')
    try:
        a = float(str(age).replace(',', '.')) if isinstance(age, str) else float(age)
    except ValueError:
        raise CollectError(f'возраст не число: {age!r}') from None
    if not 0 <= a <= 120:
        raise CollectError(f'возраст вне диапазона 0–120: {a:g}')
    return a


def subject_meta(pid, hb, unit, sex, age, pregnant, eye, lab_gap_days, consent):
    """Проверка анкеты → dict для hb.csv. Ошибка → CollectError с текстом для оператора."""
    if consent is not True:
        raise CollectError('нет отметки о письменном согласии участника — без него фото не сохраняется')
    pid = check_id(pid)
    s = {'M': 'M', 'М': 'M', 'F': 'F', 'Ж': 'F'}.get(str(sex or '').strip().upper()[:1])
    if s is None:
        raise CollectError('укажите пол (М / Ж): от него зависит норма ВОЗ')
    a = parse_age(age)
    if a < CHILD_MIN_AGE:
        raise CollectError(f'детей младше {CHILD_MIN_AGE} лет не собираем: модели для них нет, '
                           'оценка по фото не проводится')
    try:
        hb_g_l = hb_to_g_l(hb, unit)
    except ValueError as e:
        raise CollectError(f'Hb: {e}') from None
    p = norm_pregnant(pregnant)
    if s == 'M' and p:
        raise CollectError('беременность отмечена у мужчины — проверьте пол')
    if eye not in EYES:
        raise CollectError('укажите глаз: L (левый) или R (правый)')
    try:
        gap = int(lab_gap_days)
    except (TypeError, ValueError):
        raise CollectError('укажите, сколько дней между анализом крови и фото') from None
    if not 0 <= gap <= MAX_LAB_GAP_DAYS:
        raise CollectError(f'анализ крови и фото разнесены на {gap} дн.: нужно не больше {MAX_LAB_GAP_DAYS} '
                           '(Hb меняется, метка будет неверной)')
    return {'patient_id': pid, 'hb_g_l': round(hb_g_l, 1), 'sex': s, 'age': a,
            'pregnant': '' if p is None else p, 'eye': eye, 'lab_gap_days': gap}


def check_photo(rgb):
    """Проверка кадра по протоколу теми же функциями, что в ядре. → dict:
    ok (можно сохранять), blocking (почему нельзя), warnings, blur (резкость, ≥ 1,3 — годно),
    conj (bool-маска предложения конъюнктивы в размере rgb; может быть пустой — тогда рисуется вручную)."""
    blocking = list(frame_issues(rgb))
    warnings = []
    h, w = rgb.shape[:2]
    conj = np.zeros((h, w), bool)
    blur = float('nan')
    if not blocking:
        det = R.detect_regions(rgb, mode='auto', landmarker=None)   # без лица: протокол — глаз крупно
        crop, roi = det['crop'], det['conj'] | det['sclera']
        issues = R.quality(crop, det['conj'], det['sclera'], det['src_side'])
        blur = R.blur_ratio(crop, roi if roi.any() else np.ones(crop.shape[:2], bool))
        if not roi.any():
            issues = [i for i in issues if i != 'не видно ни конъюнктивы, ни склеры']
            if blur < 1.3:
                issues.append('снимок размыт')
            warnings.append('конъюнктива и склера автоматически не найдены — обведите конъюнктиву кистью')
        elif not det['conj'].any():
            warnings.append('конъюнктива автоматически не найдена — обведите её кистью')
        blocking += [i for i in issues if i in BLOCKING]
        warnings += [i for i in issues if i not in BLOCKING] + list(det['notes'])
        if det['conj'].any():
            conj = resize_to(det['conj'], (h, w))     # вырезка — весь кадр (лицо не ищем), масштаб тот же
    return {'ok': not blocking, 'blocking': blocking, 'warnings': warnings, 'blur': blur, 'conj': conj}


def mask_from_layers(layers, shape):
    """Слои редактора (RGBA, нарисованное — непрозрачное) → bool-маска размера shape (h, w)."""
    m = np.zeros(shape, bool)
    for lay in layers or []:
        a = np.asarray(lay)
        if a.ndim != 3 or a.shape[2] != 4:
            continue
        m |= resize_to(a[..., 3] > 0, shape)
    return m


def outline(img_u8, mask, color=(0, 229, 255), width=2):
    """Контур предложенной маски поверх фото — только подсказка в редакторе (сохраняется исходный снимок)."""
    out = np.ascontiguousarray(img_u8.copy())
    cs, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, cs, -1, color, width)
    return out


MASK_CHOICES = ('draw', 'auto', 'none')


def choose_mask(drawn, auto, choice):
    """Какая маска сохраняется → (mask | None, источник).
    draw — только нарисованное кистью; auto — предложение (+ дорисованное кистью); none — без маски."""
    if choice == 'none':
        return None, ''
    if choice == 'draw':
        if not drawn.any():
            raise CollectError('маска пустая: закрасьте конъюнктиву кистью или выберите другой вариант маски')
        return drawn, 'manual'
    if choice == 'auto':
        if auto is None or not auto.any():
            raise CollectError('предложенной маски нет: закрасьте конъюнктиву кистью')
        return (auto | drawn, 'auto+manual') if drawn.any() else (auto, 'auto')
    raise CollectError(f'вариант маски: {", ".join(MASK_CHOICES)}')


def _png(path, arr):
    Image.fromarray(arr).save(path, format='PNG')     # PIL не пишет EXIF/GPS в PNG, если их не передать


def _next_index(img_dir, pid, eye):
    n = 1
    while os.path.exists(os.path.join(img_dir, f'{pid}_{eye}_{n}.png')):
        n += 1
    return n


def save_record(root, rgb, meta, mask=None, mask_source='', blur=float('nan')):
    """Сохранить снимок (PNG без метаданных, ≤ MAX_SAVE_SIDE) и маску в <root>/phone/, строку — в hb.csv.
    rgb — float [0, 1] уже повёрнутый по EXIF (predict.load_photo); meta — из subject_meta. → dict путей."""
    d = os.path.join(root, 'phone')
    img_dir, mask_dir = os.path.join(d, 'images'), os.path.join(d, 'masks')
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    small = resize_max(np.clip(rgb, 0, 1).astype(np.float32), MAX_SAVE_SIDE)
    pid, eye = meta['patient_id'], meta['eye']
    name = f'{pid}_{eye}_{_next_index(img_dir, pid, eye)}.png'
    img_rel = f'images/{name}'
    _png(os.path.join(d, img_rel), (small * 255 + 0.5).astype(np.uint8))
    mask_rel = ''
    if mask is not None and mask.any():
        m = resize_to(mask, small.shape[:2])
        rgba = np.zeros((*m.shape, 4), np.uint8)
        rgba[m] = (255, 255, 255, 255)              # прозрачный фон, ткань непрозрачна — так читает io.load_mask
        mask_rel = f'masks/{name}'
        _png(os.path.join(d, mask_rel), rgba)
    row = {**meta, 'file': img_rel, 'mask_conj': mask_rel, 'mask_source': mask_source if mask_rel else '',
           'blur': '' if np.isnan(blur) else round(blur, 2)}
    path = os.path.join(d, 'hb.csv')
    new = not os.path.exists(path)
    with open(path, 'a', newline='', encoding='utf-8') as f:
        wr = csv.DictWriter(f, HB_COLUMNS)
        if new:
            wr.writeheader()
        wr.writerow({k: row[k] for k in HB_COLUMNS})
    return {'image': os.path.join(d, img_rel), 'mask': os.path.join(d, mask_rel) if mask_rel else None,
            'hb_csv': path, 'file': img_rel}


def known_ids(root):
    path = os.path.join(root, 'phone', 'hb.csv')
    if not os.path.exists(path):
        return set()
    with open(path, encoding='utf-8') as f:
        return {r['patient_id'] for r in csv.DictReader(f)}


def summary(root):
    """Сколько собрано: снимков, участников, с маской, с анемией по ВОЗ (по лабораторному Hb)."""
    from .labs import is_anemic
    path = os.path.join(root, 'phone', 'hb.csv')
    if not os.path.exists(path):
        return {'photos': 0, 'patients': 0, 'with_mask': 0, 'anemic_patients': 0}
    with open(path, encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    pats = {r['patient_id']: r for r in rows}
    anemic = sum(is_anemic(float(r['hb_g_l']), r['sex'], float(r['age']), norm_pregnant(r['pregnant']))
                 for r in pats.values())
    return {'photos': len(rows), 'patients': len(pats), 'with_mask': sum(bool(r['mask_conj']) for r in rows),
            'anemic_patients': int(anemic)}


def load_for_collect(src):
    """Файл/массив → RGB float [0, 1] с поворотом по EXIF; оригинал (и его EXIF) дальше не используется."""
    return load_photo(src)


__all__ = [
    'PROTOCOL',
    'CollectError',
    'check_id',
    'check_photo',
    'known_ids',
    'load_for_collect',
    'mask_from_layers',
    'choose_mask',
    'outline',
    'new_id',
    'save_record',
    'subject_meta',
    'summary',
]
