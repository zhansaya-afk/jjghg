"""Сбор своих фото со смартфона + лабораторный Hb того же дня → data/phone/ (формат anemia.data.load_phone).

Логика без интерфейса (интерфейс — app/collect.py). Правила:
- id участника — псевдоним 'P' + 6 hex-символов, генерируется случайно (данные псевдонимизированы, а не обезличены:
  фото глаза — биометрический признак); ФИО, телефон, имя исходного файла нигде не сохраняются; EXIF/GPS/модель
  телефона отбрасываются (пиксели пересохраняются в PNG без метаданных).
- Hb — из бланка анализа (labs.hb_to_g_l), хранится в hb.csv; оценка модели по фото здесь не считается и не пишется.
- Пилот, этап 1 (протокол v1.1): запись без Hb (photo_meta, save_pilot_record → <root>/pilot/photos.csv); Hb из ЛИС
  клиники сопоставляет по коду только врач-исследователь (merge_lis, scripts/merge_lis.py).
- Дети младше 5 лет — отказ, как в ядре (predict.child_refused): модели для них нет.
- Проверка снимка — те же функции, что в ядре: predict.frame_issues (кадр) и regions.quality (резкость, свет,
  размер ткани). Предложение маски конъюнктивы — regions.detect_regions (без обучения), дальше правка кистью.
"""
import csv
import math
import os
import re
import secrets
import shutil
import stat
import time

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
# необязательные поля (2026-11): только исходные значения анкеты, пусто = не указано. Поправки на высоту и курение
# по ним НЕ считаются и метки не меняют — это отдельное медицинское решение. Новые колонки — в конце: старые
# файлы hb.csv читаются как раньше, при первой записи к ним добавляются пустые колонки (копия — hb.csv.bak или hb.csv.bak-<дата-время>).
OPTIONAL_COLUMNS = ['trimester', 'residence_city', 'elevation_m', 'smoking', 'cigarettes_per_day']
HB_COLUMNS_ALL = HB_COLUMNS + OPTIONAL_COLUMNS
SMOKING = ('no', 'yes')
CITY_RE = re.compile(r"^[A-Za-zА-Яа-яЁёӘәҒғҚқҢңӨөҰұҮүҺһІі .'-]{1,40}$")
ELEVATION_RANGE = (-500.0, 9000.0)   # м: проверка ввода (опечатки), не медицинское правило
CIGARETTES_RANGE = (1.0, 100.0)      # сигарет в день при «курит»: проверка ввода (опечатки)

PROTOCOL = """**Протокол съёмки**
1. Анализ крови (Hb) — в тот же день, что и фото. Нужно письменное согласие участника.
2. Свет: дневной у окна, без прямого солнца; **без вспышки**, без фильтров и «улучшайзеров» камеры.
3. Оттяните нижнее веко пальцем, взгляд вверх: розовая слизистая должна быть видна полосой.
4. Телефон в 10–15 см от глаза, фокус — тапом по веку. Веко занимает заметную часть кадра.
5. По 2 снимка каждого глаза. Лицо целиком не нужно: только глаз крупно."""


class CollectError(ValueError):
    """Запись нельзя сохранить: понятное сообщение для оператора."""


def new_id(existing=()):
    """Случайный псевдонимный id участника ('P3fa9c1'), не совпадающий с уже существующими."""
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


def subject_meta(pid, hb, unit, sex, age, pregnant, eye, lab_gap_days, consent, optional=None):
    """Проверка анкеты → dict для hb.csv. Ошибка → CollectError с текстом для оператора.
    optional — необязательные поля (optional_meta): в dict попадают только указанные."""
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
    meta = {'patient_id': pid, 'hb_g_l': round(hb_g_l, 1), 'sex': s, 'age': a,
            'pregnant': '' if p is None else p, 'eye': eye, 'lab_gap_days': gap}
    extra = optional_meta(p, **(optional or {}))
    return {**meta, **{k: v for k, v in extra.items() if v != ''}}


def _blank(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def _number(v, what):
    try:
        return float(str(v).replace(',', '.')) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        raise CollectError(f'{what}: не число ({v!r})') from None


def optional_meta(pregnant=None, trimester=None, residence_city=None, elevation_m=None, smoking=None,
                  cigarettes_per_day=None):
    """Необязательные поля анкеты → dict исходных значений ('' = не указано). Ничего не подставляется и не
    пересчитывается; противоречие (триместр без беременности, сигареты у некурящего) → CollectError."""
    out = dict.fromkeys(OPTIONAL_COLUMNS, '')
    if not _blank(trimester):
        if pregnant is not True:
            raise CollectError('триместр указывается только при беременности')
        t = _number(trimester, 'триместр')
        if t not in (1, 2, 3):
            raise CollectError(f'триместр — 1, 2 или 3, а не {trimester!r}')
        out['trimester'] = int(t)
    if not _blank(residence_city):
        c = str(residence_city).strip()
        if not CITY_RE.match(c):
            raise CollectError('город — только название (буквы, пробел, дефис, до 40 знаков), без имён и адресов')
        out['residence_city'] = c
    if not _blank(elevation_m):
        e = _number(elevation_m, 'высота над уровнем моря')
        if not ELEVATION_RANGE[0] <= e <= ELEVATION_RANGE[1]:
            raise CollectError(f'высота {e:g} м вне диапазона {ELEVATION_RANGE[0]:g}…{ELEVATION_RANGE[1]:g} м')
        out['elevation_m'] = e
    if not _blank(smoking):
        if smoking not in SMOKING:
            raise CollectError(f'курение: {", ".join(SMOKING)} или не указано')
        out['smoking'] = smoking
    if not _blank(cigarettes_per_day):
        if out['smoking'] != 'yes':
            raise CollectError('сигарет в день указывается только при ответе «курит»')
        n = _number(cigarettes_per_day, 'сигарет в день')
        if not CIGARETTES_RANGE[0] <= n <= CIGARETTES_RANGE[1]:
            raise CollectError(f'сигарет в день: {n:g} вне диапазона {CIGARETTES_RANGE[0]:g}–{CIGARETTES_RANGE[1]:g}')
        out['cigarettes_per_day'] = n
    return out


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
    path = os.path.join(d, 'hb.csv')
    _check_writable(path)          # до записи снимка: без прав на запись ничего не сохраняется
    if os.path.exists(path):
        _upgrade_header(path)      # до записи снимка: при неожиданном заголовке ничего не сохраняется
    out = _write_record(d, path, HB_COLUMNS_ALL, rgb, meta, mask, mask_source, blur)
    return {'image': out['image'], 'mask': out['mask'], 'hb_csv': path, 'file': out['file']}


def _write_record(d, path, columns, rgb, meta, mask, mask_source, blur):
    """Снимок + маска в d/images, d/masks; строка (только columns) — в CSV path. Общая часть save_record и
    save_pilot_record."""
    try:
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
        new = not os.path.exists(path)
        with open(path, 'a', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, columns)
            if new:
                wr.writeheader()
            wr.writerow({k: row.get(k, '') for k in columns})
    except OSError as e:          # диск, права, сеть: понятное сообщение для оператора С1
        name = os.path.basename(path)
        raise CollectError(f'запись не удалась ({e.strerror or e}); строки {name}, записанные раньше, не изменены') from e
    return {'image': os.path.join(d, img_rel), 'mask': os.path.join(d, mask_rel) if mask_rel else None,
            'csv': path, 'file': img_rel}


def _read_only(p):
    """Нет прав на запись: для текущего пользователя (os.access) ИЛИ ни одного бита записи у файла/папки.
    Второе нужно для root (Colab): root обходит права, и os.access у него всегда True."""
    return not os.access(p, os.W_OK) or not os.stat(p).st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)


def _check_writable(path):
    """Можно ли безопасно писать hb.csv: файл и папка доступны на запись (иначе — до любых изменений)."""
    d = os.path.dirname(path)
    if os.path.exists(path) and _read_only(path):
        raise CollectError(f'{path}: файл только для чтения — запись остановлена, данные не изменены')
    if os.path.isdir(d) and _read_only(d):
        raise CollectError(f'{d}: папка недоступна на запись — запись остановлена, данные не изменены')


def _read_hb_csv(path):
    """hb.csv → (заголовок, строки). Пустой файл, пустой заголовок, лишние или недостающие ячейки → CollectError."""
    with open(path, newline='', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        head, rows = list(rd.fieldnames or []), list(rd)
    if not head:
        raise CollectError('hb.csv пуст или без заголовка; файл не изменён')
    for i, r in enumerate(rows, start=2):
        if None in r:
            raise CollectError(f'hb.csv, строка {i}: лишние ячейки ({len(head) + len(r[None])} вместо '
                               f'{len(head)}); файл не изменён')
        if any(v is None for v in r.values()):
            raise CollectError(f'hb.csv, строка {i}: не хватает ячеек; файл не изменён')
    return head, rows


def _backup_path(path):
    """Имя резервной копии, которое ещё не занято: hb.csv.bak, иначе hb.csv.bak-ГГГГММДД-ЧЧММСС[-N]."""
    if not os.path.exists(path + '.bak'):
        return path + '.bak'
    base = path + '.bak-' + time.strftime('%Y%m%d-%H%M%S')
    cand, n = base, 1
    while os.path.exists(cand):
        cand, n = f'{base}-{n}', n + 1
    return cand


def _upgrade_header(path):
    """Старый hb.csv (без необязательных колонок) → те же строки и значения + пустые новые колонки в конце.
    Порядок: проверка прав и структуры → резервная копия под незанятым именем (существующие копии не
    перезаписываются) → новый файл рядом → сверка значений → атомарная замена. Любая ошибка → CollectError,
    исходный файл не изменён. Уже расширенный файл не трогается."""
    _check_writable(path)
    head, rows = _read_hb_csv(path)
    if head == HB_COLUMNS_ALL:
        return
    if head != HB_COLUMNS_ALL[:len(head)] or len(head) < len(HB_COLUMNS):
        raise CollectError(f'hb.csv: неожиданные колонки {head}; файл не изменён')
    tmp = path + '.tmp'
    try:
        bak = _backup_path(path)
        with open(path, 'rb') as src, open(bak, 'xb') as dst:     # 'x': никогда не перезаписывает
            shutil.copyfileobj(src, dst)
        with open(tmp, 'w', newline='', encoding='utf-8') as f:
            wr = csv.DictWriter(f, HB_COLUMNS_ALL)
            wr.writeheader()
            for r in rows:
                wr.writerow({k: r.get(k, '') for k in HB_COLUMNS_ALL})
            f.flush()
            os.fsync(f.fileno())
        new_head, new_rows = _read_hb_csv(tmp)
        same = new_head == HB_COLUMNS_ALL and len(new_rows) == len(rows) and all(
            all(n[k] == o[k] for k in head) and all(n[k] == '' for k in HB_COLUMNS_ALL[len(head):])
            for n, o in zip(new_rows, rows))
        if not same:
            raise CollectError('hb.csv: при обновлении заголовка значения не совпали; файл не изменён')
        shutil.copymode(path, tmp)
        os.replace(tmp, path)
    except OSError as e:
        raise CollectError(f'hb.csv: не удалось обновить ({e.strerror or e}); файл не изменён') from e
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


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


# ------------------------------------------------------------------ пилот, этап 1: запись без Hb (протокол v1.1)
# Ученики вносят снимок и анкету БЕЗ Hb: Hb этапа 1 врач-исследователь выгружает из лабораторной системы клиники
# (ЛИС) в свою таблицу hb_lis.csv и сам сопоставляет её с фото по коду (merge_lis). В папке учеников Hb нет вообще.
# Модель, пороги, признаки и метки здесь не участвуют: это только запись данных.
PILOT_DIR = 'pilot'
PILOT_CSV = 'photos.csv'
PILOT_COLUMNS = [c for c in HB_COLUMNS_ALL if c not in ('hb_g_l', 'lab_gap_days')]
PILOT_MIN_AGE = 18                 # критерий включения протокола (п. 3.1), не медицинский порог
HB_FIELDS = ('hb', 'hb_g_l', 'lab_gap_days')
LIS_REQUIRED = ('patient_id', 'hb_g_l', 'lab_gap_days')

PILOT_PROTOCOL = """**Протокол съёмки пилота (v1.1, п. 7)**
1. Одна модель телефона на весь этап; HDR, «улучшение», фильтры, портрет, зум и **вспышка выключены**.
2. Свет: одна лампа, нейтральный белый LED 4000–5000 K, шторы закрыты. Тип лампы — в анкету.
3. Участник смотрит вверх и **сам** чистым пальцем оттягивает нижнее веко; глаза не касаемся.
4. Телефон в 10–15 см, фокус — тапом по веку, только глаз крупно. Правый глаз (нельзя — левый, отметить).
5. Снимки 1 и 2 — стандартный свет (между ними веко опустить и оттянуть снова); снимок 3 — свет помещения.
6. Не больше 3 попыток на снимок; не вышло — это отказ, запишите его в анкету.
7. Результат программы участнику не показывается и здесь не считается."""


def pilot_eligibility(age, pregnant):
    """Критерии протокола пилота: только взрослые 18+, беременные не включаются. → возраст (float)."""
    a = parse_age(age)
    if a < PILOT_MIN_AGE:
        raise CollectError(f'в пилот включаются только взрослые {PILOT_MIN_AGE}+ (протокол v1.1, п. 3.1)')
    if norm_pregnant(pregnant):
        raise CollectError('беременные в пилот не включаются (протокол v1.1, п. 3.2)')
    return a


def photo_meta(pid, sex, age, pregnant, eye, consent, optional=None):
    """Анкета пилота (этап 1) без Hb → dict для photos.csv. Hb сюда не передаётся вовсе."""
    if consent is not True:
        raise CollectError('нет отметки о письменном согласии участника — без него фото не сохраняется')
    pid = check_id(pid)
    s = {'M': 'M', 'М': 'M', 'F': 'F', 'Ж': 'F'}.get(str(sex or '').strip().upper()[:1])
    if s is None:
        raise CollectError('укажите пол (М / Ж): от него зависит норма ВОЗ')
    a = pilot_eligibility(age, pregnant)
    if eye not in EYES:
        raise CollectError('укажите глаз: L (левый) или R (правый)')
    extra = optional_meta(False, **(optional or {}))
    meta = {'patient_id': pid, 'sex': s, 'age': a, 'pregnant': False, 'eye': eye}
    return {**meta, **{k: v for k, v in extra.items() if v != ''}}


def pilot_paths(root):
    d = os.path.join(root, PILOT_DIR)
    return d, os.path.join(d, PILOT_CSV)


def save_pilot_record(root, rgb, meta, mask=None, mask_source='', blur=float('nan')):
    """Снимок + анкета без Hb → <root>/pilot/images, masks, photos.csv. Hb в meta → CollectError (ничего не пишется)."""
    leaked = [k for k in HB_FIELDS if k in meta]
    if leaked:
        raise CollectError(f'в записи пилота не должно быть Hb ({", ".join(leaked)}): его вносит врач-исследователь '
                           'из ЛИС отдельно; ничего не сохранено')
    d, path = pilot_paths(root)
    _check_writable(path)
    if os.path.exists(path):
        head, _ = _read_hb_csv(path)
        if head != PILOT_COLUMNS:
            raise CollectError(f'{PILOT_CSV}: неожиданные колонки {head}; ничего не сохранено')
    return _write_record(d, path, PILOT_COLUMNS, rgb, meta, mask, mask_source, blur)


def _pilot_rows(root):
    _, path = pilot_paths(root)
    if not os.path.exists(path):
        return []
    return _read_hb_csv(path)[1]


def pilot_known_ids(root):
    return {r['patient_id'] for r in _pilot_rows(root)}


def pilot_summary(root):
    """Сколько собрано на этапе 1: снимков, участников, с маской. Ни Hb, ни доли анемии здесь нет."""
    rows = _pilot_rows(root)
    return {'photos': len(rows), 'patients': len({r['patient_id'] for r in rows}),
            'with_mask': sum(bool(r['mask_conj']) for r in rows)}


def merge_lis(photos_csv, lis_csv, out_csv):
    """Для врача-исследователя: photos.csv (без Hb) + hb_lis.csv из ЛИС → hb.csv в формате load_phone.

    hb_lis.csv: patient_id, hb_g_l, lab_gap_days (+ необязательно unit, blood_time, analyzer — не копируются).
    Пути к снимкам пишутся относительно папки out_csv. out_csv не перезаписывается и не может лежать в папке
    учеников (рядом с photos.csv). Участники без Hb или с анализом не в тот же день (> MAX_LAB_GAP_DAYS) в hb.csv
    не попадают, но возвращаются списками: их нужно отчитать (STARD), а не потерять молча.
    Чтобы load_phone прочитал результат, out_csv должен называться <папка врача>/phone/hb.csv.
    → {'merged': снимков, 'patients': участников, 'no_hb': [...], 'lab_gap_too_large': [...], 'lis_without_photo': [...]}"""
    photos_dir = os.path.dirname(os.path.abspath(photos_csv))
    out_dir = os.path.dirname(os.path.abspath(out_csv))
    students_root = os.path.realpath(os.path.dirname(photos_dir))     # <data> учеников: pilot/ лежит внутри
    real_out = os.path.realpath(out_dir)
    if os.path.commonpath([real_out, students_root]) == students_root:
        raise CollectError('hb.csv с Hb нельзя класть в папку данных учеников (ни в pilot/, ни рядом): '
                           'укажите папку врача')
    if os.path.exists(out_csv):
        raise CollectError(f'{out_csv} уже есть — не перезаписываю; укажите новый файл')
    head, photos = _read_hb_csv(photos_csv)
    if head != PILOT_COLUMNS:
        raise CollectError(f'{PILOT_CSV}: неожиданные колонки {head}')
    lis_head, lis_rows = _read_hb_csv(lis_csv)
    missing = [c for c in LIS_REQUIRED if c not in lis_head]
    if missing:
        raise CollectError(f'hb_lis.csv: нет колонок {missing}')
    lis, seen = {}, set()
    for i, r in enumerate(lis_rows, start=2):
        pid = check_id(r['patient_id'])
        if pid in seen:
            raise CollectError(f'hb_lis.csv, строка {i}: код {pid} повторяется — оставьте одно значение')
        seen.add(pid)
        if _blank(r['hb_g_l']):
            continue              # нет Hb: участник попадёт в no_hb (если есть снимок)
        try:
            hb = hb_to_g_l(r['hb_g_l'], (r.get('unit') or '').strip() or 'г/л')
        except ValueError as e:
            raise CollectError(f'hb_lis.csv, строка {i}: Hb: {e}') from None
        try:
            gap = int(r['lab_gap_days'])
        except (TypeError, ValueError):
            raise CollectError(f'hb_lis.csv, строка {i}: lab_gap_days — целое число дней') from None
        lis[pid] = (round(hb, 1), gap)
    try:
        rel = os.path.relpath(photos_dir, out_dir)
    except ValueError:            # Windows: другой диск — относительного пути нет, пишем абсолютный
        rel = photos_dir
    out_rows, no_hb, far = [], set(), set()
    for r in photos:
        pid = r['patient_id']
        if pid not in lis:
            no_hb.add(pid)
            continue
        hb, gap = lis[pid]
        if not 0 <= gap <= MAX_LAB_GAP_DAYS:
            far.add(pid)
            continue
        row = {**r, 'hb_g_l': hb, 'lab_gap_days': gap,
               'file': os.path.join(rel, r['file']).replace(os.sep, '/'),
               'mask_conj': os.path.join(rel, r['mask_conj']).replace(os.sep, '/') if r['mask_conj'] else ''}
        out_rows.append(row)
    os.makedirs(out_dir, exist_ok=True)
    with open(out_csv, 'x', newline='', encoding='utf-8') as f:       # 'x': никогда не перезаписывает
        wr = csv.DictWriter(f, HB_COLUMNS_ALL)
        wr.writeheader()
        for row in out_rows:
            wr.writerow({k: row.get(k, '') for k in HB_COLUMNS_ALL})
    photo_ids = {r['patient_id'] for r in photos}
    return {'merged': len(out_rows), 'patients': len({r['patient_id'] for r in out_rows}), 'no_hb': sorted(no_hb), 'lab_gap_too_large': sorted(far),
            'lis_without_photo': sorted(seen - photo_ids)}


def load_for_collect(src):
    """Файл/массив → RGB float [0, 1] с поворотом по EXIF; оригинал (и его EXIF) дальше не используется."""
    return load_photo(src)


__all__ = [
    'PILOT_PROTOCOL',
    'PROTOCOL',
    'CollectError',
    'check_id',
    'check_photo',
    'known_ids',
    'load_for_collect',
    'mask_from_layers',
    'merge_lis',
    'photo_meta',
    'pilot_eligibility',
    'pilot_known_ids',
    'pilot_summary',
    'choose_mask',
    'outline',
    'new_id',
    'optional_meta',
    'save_pilot_record',
    'save_record',
    'subject_meta',
    'summary',
]
