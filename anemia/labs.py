"""Нормы по гемоглобину. Все значения Hb в г/л.

who_limits / who_grade / is_anemic — нормы ВОЗ 2011 (WHO, Haemoglobin concentrations for the diagnosis of anaemia and
assessment of severity, 2011, таблица 1). Ими пользуются фотоанализ, метки датасетов и персонализатор: не менять.

Готовый анализ крови в ячейке Я3 (lab_report) классифицируется двумя отдельными системами, которые не смешиваются:
- classify_who2024 — WHO, Guideline on haemoglobin cutoffs to define anaemia in individuals and populations (2024),
  таблицы 2–3 (https://www.ncbi.nlm.nih.gov/books/NBK602198/);
- classify_kz2017 — клинический протокол МЗ РК «Железодефицитная анемия» (взрослые), 2017, протокол № 29
  (https://diseases.medelement.com/disease/15415).
Поправки ВОЗ 2024 на высоту проживания и курение здесь не применяются (нужно отдельное медицинское согласование).
lab_hb_report — прежний отчёт по ВОЗ 2011 без изменений (интерфейс HemaVision и персонализатор).
"""
import math
import re
from dataclasses import dataclass

GRADES = ('norm', 'mild', 'moderate', 'severe')
GRADE_RU = {'norm': 'Норма', 'mild': 'Лёгкая анемия', 'moderate': 'Умеренная анемия', 'severe': 'Тяжёлая анемия'}


@dataclass(frozen=True)
class WhoLimits:
    normal: float    # Hb ≥ normal → норма
    mild: float      # mild ≤ Hb < normal → лёгкая
    moderate: float  # moderate ≤ Hb < mild → умеренная; ниже → тяжёлая


def norm_pregnant(v):
    """Статус беременности из источника → True / False / None (неизвестно). Только из явного поля данных:
    из пола, возраста или других признаков не выводится; пусто, NaN или непонятное значение → None."""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, float) and v != v:
        return None
    s = str(v).strip().lower()
    if s in ('1', '1.0', 'true', 'yes', 'y', 'да'):
        return True
    if s in ('0', '0.0', 'false', 'no', 'n', 'нет'):
        return False
    return None


def is_pregnant(v):
    """Для выбора нормы ВОЗ: беременность только подтверждённая. Неизвестный статус (None) → нормы небеременных,
    как и раньше (пороги не меняются), но сам статус в данных остаётся None и в False не превращается."""
    return norm_pregnant(v) is True


def who_limits(sex=None, age=None, pregnant=None):
    """sex: 'M' / 'F' / None, age в годах (дробный для детей до 5 лет), pregnant: True / False / None (неизвестно)."""
    if is_pregnant(pregnant) or (age is not None and age < 5):
        return WhoLimits(110, 100, 70)
    if age is not None and age < 12:
        return WhoLimits(115, 110, 80)
    if age is not None and age < 15:
        return WhoLimits(120, 110, 80)
    if sex == 'M':
        return WhoLimits(130, 110, 80)
    return WhoLimits(120, 110, 80)


def who_grade(hb_g_l, sex=None, age=None, pregnant=None):
    lim = who_limits(sex, age, pregnant)
    if hb_g_l >= lim.normal:
        return 'norm'
    if hb_g_l >= lim.mild:
        return 'mild'
    if hb_g_l >= lim.moderate:
        return 'moderate'
    return 'severe'


def is_anemic(hb_g_l, sex=None, age=None, pregnant=None):
    return hb_g_l < who_limits(sex, age, pregnant).normal


# ------------------------------------------------------------------ готовый анализ крови (отдельно от фото)
HB_UNITS = {'g/l': 1.0, 'г/л': 1.0, 'g/dl': 10.0, 'г/дл': 10.0, 'g%': 10.0, 'г%': 10.0,
            'mmol/l': 16.11, 'ммоль/л': 16.11}       # 1 ммоль/л Hb (мономер, Fe) = 16,11 г/л
HB_PLAUSIBLE = (30.0, 250.0)                         # г/л: вне диапазона — почти наверняка не та единица


def hb_to_g_l(value, unit=None):
    """Hb из бланка анализа → г/л. value — число или строка («12,5 г/дл», «125»), unit — 'г/л', 'g/dL', 'ммоль/л'.
    Без единицы: число ≥ 30 считается г/л; меньшее неоднозначно (г/дл и ммоль/л пересекаются) → ошибка."""
    if isinstance(value, str):
        m = re.match(r'\s*([0-9]+(?:[.,][0-9]+)?)\s*(.*)$', value)
        if not m:
            raise ValueError(f'не понял значение Hb: {value!r}')
        value, unit = float(m.group(1).replace(',', '.')), unit or (m.group(2).strip() or None)
    value = float(value)
    if unit is None:
        if value < HB_PLAUSIBLE[0]:
            raise ValueError(f'Hb = {value}: укажите единицу (г/л, г/дл или ммоль/л)')
        k = 1.0
    else:
        u = unit.strip().lower().replace(' ', '')
        if u not in HB_UNITS:
            raise ValueError(f'неизвестная единица Hb: {unit!r}; допустимо: {", ".join(sorted(set(HB_UNITS)))}')
        k = HB_UNITS[u]
    g_l = value * k
    if not HB_PLAUSIBLE[0] <= g_l <= HB_PLAUSIBLE[1]:
        raise ValueError(f'Hb = {g_l:.0f} г/л вне правдоподобного диапазона {HB_PLAUSIBLE[0]:.0f}–'
                         f'{HB_PLAUSIBLE[1]:.0f} г/л: проверьте число и единицу')
    return g_l


# ------------------------------------------------------------------ анализ крови: ВОЗ 2024 и протокол МЗ РК 2017
WHO2024_SOURCE = 'ВОЗ 2024 (Guideline on haemoglobin cutoffs to define anaemia, табл. 2–3)'
KZ2017_SOURCE = 'протокол МЗ РК «Железодефицитная анемия» (взрослые), 2017, № 29'
WHO2024_MIN_AGE = 0.5        # лет: младше 6 месяцев ВОЗ 2024 порогов не даёт
WHO2024_ADULT_MAX_AGE = 65   # лет: пороги взрослых в ВОЗ 2024 даны для 15–65 лет
KZ_MIN_AGE = 18              # лет: протокол МЗ РК — для взрослых
MIN_PREGNANCY_AGE = 10       # лет: беременность младше — почти наверняка ошибка в возрасте (проверка ввода)
TRIMESTERS = (1, 2, 3)
KZ_GRADE_RU = {'norm': 'Нет анемии', 'mild': 'Анемия лёгкой степени', 'moderate': 'Анемия средней степени',
               'severe': 'Анемия тяжёлой степени'}
# Статус железодефицита — отдельно от степени анемии: шкала РК оценивает только Hb, а диагноз ЖДА по протоколу
# требует показателей обмена железа. Я3 их не принимает, поэтому железодефицит здесь всегда не подтверждён.
IRON_NOT_CONFIRMED = ('Железодефицит не подтверждён: недостаточно лабораторных данных '
                      '(нет ферритина и насыщения трансферрина, TSat).')
NO_ADJUSTMENTS = 'Поправки на высоту проживания и курение не применены.'
NO_SMOKING_ADJUSTMENT = 'Поправка на курение не применяется (поправки на высоту и курение не суммируются).'
# Поправка ВОЗ 2024 на высоту проживания (таблица 4, NBK602198): (верхняя граница высоты, м, не включая; г/л).
# Прибавляется к порогам (то же, что вычесть из измеренного Hb). Выше 2999 м таблица в модуле не задана:
# отказ с сообщением, а не догадка. Только по введённой высоте; без высоты поправки нет.
WHO2024_ALTITUDE = ((500, 0), (1000, 4), (1500, 8), (2000, 11), (2500, 14), (3000, 18))
ELEVATION_INPUT_RANGE = (-500.0, 9000.0)   # м: проверка ввода (опечатки)
NOT_DIAGNOSIS = 'Это интерпретация анализа крови, не оценка по фото и не диагноз; диагноз ставит врач.'

# группа → (подпись, норма от, лёгкая от, умеренная от); ниже «умеренная от» — тяжёлая. ВОЗ 2024, таблица 3.
WHO2024_GROUPS = {
    'child_6_23m': ('дети 6–23 мес', 105, 95, 70),
    'child_24_59m': ('дети 24–59 мес', 110, 100, 70),
    'child_5_11': ('дети 5–11 лет', 115, 110, 80),
    'child_12_14': ('дети 12–14 лет', 120, 110, 80),
    'women': ('женщины 15–65 лет, не беременные', 120, 110, 80),
    'men': ('мужчины 15–65 лет', 130, 110, 80),
    'preg_1': ('беременные, I триместр', 110, 100, 70),
    'preg_2': ('беременные, II триместр', 105, 95, 70),
    'preg_3': ('беременные, III триместр', 110, 100, 70),
}


class LabDataError(ValueError):
    """Анализ крови нельзя классифицировать: данных недостаточно или они противоречат друг другу.
    .report — полный отчёт (обе системы, текст для врача), чтобы показать, чего не хватает."""

    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report


class _Invalid(ValueError):
    pass


def _blank(v):
    return v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip())


def _norm_sex(v):
    if _blank(v):
        return None
    s = {'M': 'M', 'М': 'M', 'F': 'F', 'Ж': 'F'}.get(str(v).strip().upper())
    if s is None:
        raise _Invalid(f'пол не распознан: {v!r} (нужно M / F)')
    return s


def _norm_age(v):
    if isinstance(v, bool):
        raise _Invalid(f'возраст не число: {v!r}')
    if _blank(v):
        return None
    try:
        a = float(str(v).replace(',', '.')) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        raise _Invalid(f'возраст не число: {v!r}') from None
    if not 0 <= a <= 120:
        raise _Invalid(f'возраст вне диапазона 0–120 лет: {a:g}')
    return a


def _norm_trimester(v):
    if _blank(v):
        return None
    try:
        t = float(str(v).strip()) if isinstance(v, str) else float(v)
    except (TypeError, ValueError):
        t = None
    if isinstance(v, bool) or t not in TRIMESTERS:
        raise _Invalid(f'триместр не распознан: {v!r} (нужно 1, 2 или 3)')
    return int(t)


def _inputs(sex, age, pregnant, trimester):
    """Проверка анкеты → (пол, возраст, беременность True/False/None, триместр). Противоречие → _Invalid."""
    s, a, p, t = _norm_sex(sex), _norm_age(age), norm_pregnant(pregnant), _norm_trimester(trimester)
    if p is True and s == 'M':
        raise _Invalid('беременность отмечена у мужчины: проверьте пол')
    if t is not None and p is not True:
        raise _Invalid('указан триместр, но беременность не отмечена: проверьте анкету')
    if p is True and a is not None and a < MIN_PREGNANCY_AGE:
        raise _Invalid(f'беременность отмечена в возрасте {a:g} лет: проверьте возраст')
    return s, a, p, t


def _norm_hb(hb_g_l):
    """Hb в г/л для классификаторов: пусто, не число, NaN/∞ или вне HB_PLAUSIBLE → _Invalid (не «анемия»)."""
    try:
        hb = float(hb_g_l)
    except (TypeError, ValueError):
        raise _Invalid(f'Hb не число ({hb_g_l!r})') from None
    if isinstance(hb_g_l, bool) or not math.isfinite(hb):
        raise _Invalid(f'Hb не число ({hb_g_l!r})')
    if not HB_PLAUSIBLE[0] <= hb <= HB_PLAUSIBLE[1]:
        raise _Invalid(f'Hb {hb:g} г/л вне правдоподобного диапазона {HB_PLAUSIBLE[0]:.0f}–{HB_PLAUSIBLE[1]:.0f} г/л')
    return hb


def _norm_elevation(elevation_m):
    """Высота проживания, м → float или None (не указана). Не число / вне диапазона → _Invalid."""
    if elevation_m is None:
        return None
    if isinstance(elevation_m, bool):
        raise _Invalid(f'высота не число ({elevation_m!r})')
    if isinstance(elevation_m, str):
        if not elevation_m.strip():
            return None
        elevation_m = elevation_m.strip().replace(',', '.')
    try:
        h = float(elevation_m)
    except (TypeError, ValueError):
        raise _Invalid(f'высота не число ({elevation_m!r})') from None
    if math.isnan(h):
        return None
    if not ELEVATION_INPUT_RANGE[0] <= h <= ELEVATION_INPUT_RANGE[1]:
        raise _Invalid(f'высота {h:g} м вне диапазона {ELEVATION_INPUT_RANGE[0]:g}…{ELEVATION_INPUT_RANGE[1]:g} м')
    return h


def who2024_altitude_adjustment(elevation_m):
    """Поправка ВОЗ 2024 к порогам Hb по высоте проживания (таблица 4), г/л. None — высота не указана.
    Высота ≥ 3000 м — ValueError: таблица для неё в модуле не задана (нужно сверить с ВОЗ)."""
    try:
        h = _norm_elevation(elevation_m)
    except _Invalid as e:
        raise ValueError(str(e)) from None
    if h is None:
        return None
    for upper, adj in WHO2024_ALTITUDE:
        if h < upper:
            return float(adj)
    raise ValueError(f'высота {h:g} м: поправка ВОЗ 2024 для высоты от 3000 м в модуле не задана')


def _hb_text(hb):
    return f'{hb:.2f}'.rstrip('0').rstrip('.').replace('.', ',')


def _grade_by(hb, normal, mild, moderate):
    return 'norm' if hb >= normal else 'mild' if hb >= mild else 'moderate' if hb >= moderate else 'severe'


def _who_group(sex, age, preg):
    """preg: False или номер триместра."""
    if preg:
        return f'preg_{preg}'
    if age < 2:
        return 'child_6_23m'
    if age < 5:
        return 'child_24_59m'
    if age < 12:
        return 'child_5_11'
    if age < 15:
        return 'child_12_14'
    return 'men' if sex == 'M' else 'women'


def _who_candidates(sex, age, pregnant, trimester):
    """Все группы ВОЗ 2024, совместимые с анкетой: неизвестное значение не подставляется, а перебирается."""
    sexes = [sex] if sex else (['F'] if pregnant is True else ['M', 'F'])
    out = set()
    for s in sexes:
        if s == 'M':
            opts = [False]
        elif pregnant is True:
            opts = [trimester] if trimester else list(TRIMESTERS)
        elif pregnant is False:
            opts = [False]
        else:   # беременность неизвестна: в таблице ВОЗ «небеременные девочки» выделены с 12 лет
            opts = [False, *TRIMESTERS] if age >= 12 else [False]
        out |= {_who_group(s, age, p) for p in opts}
    return sorted(out)


def _unknowns(sex, pregnant, trimester, age):
    u = []
    if sex is None and age >= 15 and pregnant is not True:
        u.append('пол')
    if pregnant is None and sex != 'M' and age >= 12:
        u.append('есть ли беременность')
    if pregnant is True and trimester is None:
        u.append('триместр')
    return u


def classify_who2024(hb_g_l, sex=None, age=None, pregnant=None, trimester=None, elevation_m=None):
    """Классификация лабораторного Hb по ВОЗ 2024 (таблицы 2–3). elevation_m — высота проживания, м: если указана,
    пороги повышаются на поправку таблицы 4 (who2024_altitude_adjustment); не указана — без поправки.
    Поправка на курение не применяется. → dict (+ altitude: высота и поправка или None):
    status: ok / insufficient (данных недостаточно) / invalid (противоречие); grade, grade_ru, anemic,
    normal_from_g_l (Hb не ниже этого — норма во всех возможных группах), group, message, notes.
    Неизвестные пол, беременность или триместр не подставляются: если при всех возможных значениях категория
    одна и та же — она выдаётся, иначе status = insufficient. Неизвестный возраст — всегда insufficient."""
    out = {'system': 'who2024', 'source': WHO2024_SOURCE, 'status': 'ok', 'grade': None, 'grade_ru': None,
           'anemic': None, 'normal_from_g_l': None, 'group': None, 'message': '', 'notes': [], 'altitude': None}
    try:
        hb = _norm_hb(hb_g_l)
        s, a, p, t = _inputs(sex, age, pregnant, trimester)
        h = _norm_elevation(elevation_m)
    except _Invalid as e:
        return {**out, 'status': 'invalid', 'message': f'Противоречивые или неверные данные: {e}.'}
    try:
        adj = who2024_altitude_adjustment(h)
    except ValueError as e:
        return {**out, 'status': 'insufficient', 'message': f'Недостаточно данных: {e}.'}
    if adj is not None:
        out['altitude'] = {'elevation_m': h, 'adjust_g_l': adj}
    shift = adj or 0.0
    if a is None:
        return {**out, 'status': 'insufficient',
                'message': 'Недостаточно данных: укажите возраст (пороги ВОЗ 2024 зависят от возраста).'}
    if a < WHO2024_MIN_AGE:
        return {**out, 'status': 'insufficient',
                'message': 'ВОЗ 2024 не даёт порогов гемоглобина для детей младше 6 месяцев.'}
    groups = _who_candidates(s, a, p, t)
    grades = {g: _grade_by(hb - shift, *WHO2024_GROUPS[g][1:]) for g in groups}   # = пороги + поправка
    if len(set(grades.values())) > 1:
        detail = '; '.join(f'{WHO2024_GROUPS[g][0]}: {GRADE_RU[k].lower()}' for g, k in grades.items())
        need = ', '.join(_unknowns(s, p, t, a)) or 'анкету'
        return {**out, 'status': 'insufficient',
                'message': f'Недостаточно данных: при Hb {_hb_text(hb)} г/л категория ВОЗ 2024 зависит от того, '
                           f'что не указано ({detail}). Укажите: {need}.'}
    grade = next(iter(grades.values()))
    notes = []
    if len(groups) == 1:
        label = WHO2024_GROUPS[groups[0]][0]
    else:
        label = f'не указано: {", ".join(_unknowns(s, p, t, a))}'
        notes.append('Категория одинакова при всех возможных значениях того, что не указано '
                     f'({"; ".join(WHO2024_GROUPS[g][0] for g in groups)}).')
    if adj is not None:
        notes.append(f'Порог с учётом высоты проживания {h:g} м: +{adj:g} г/л ко всем порогам '
                     '(ВОЗ 2024, таблица 4).' if adj else
                     f'Высота проживания {h:g} м: ниже 500 м, поправка ВОЗ 2024 не нужна (0 г/л).')
    if a > WHO2024_ADULT_MAX_AGE and not p:
        notes.append('ВОЗ 2024 даёт пороги взрослых для 15–65 лет; отдельных порогов старше 65 лет нет, '
                     'применены пороги взрослых.')
    return {**out, 'grade': grade, 'grade_ru': GRADE_RU[grade], 'anemic': grade != 'norm',
            'normal_from_g_l': float(max(WHO2024_GROUPS[g][1] for g in groups)) + shift, 'group': label,
            'notes': notes}


def who2024_normal(sex=None, age=None, pregnant=None, trimester=None, elevation_m=None):
    """Порог нормы ВОЗ 2024 для профиля, г/л (с поправкой на высоту, если она указана). Неизвестный пол /
    беременность / триместр → наибольший из возможных порогов (как normal_from_g_l в classify_who2024).
    Неизвестный или < 6 мес возраст, противоречие, высота ≥ 3000 м → None."""
    try:
        s, a, p, t = _inputs(sex, age, pregnant, trimester)
        adj = who2024_altitude_adjustment(elevation_m) or 0.0
    except (_Invalid, ValueError):
        return None
    if a is None or a < WHO2024_MIN_AGE:
        return None
    return float(max(WHO2024_GROUPS[g][1] for g in _who_candidates(s, a, p, t))) + adj


def classify_kz2017(hb_g_l, sex=None, age=None, pregnant=None):
    """Степень АНЕМИИ по шкале протокола МЗ РК «ЖДА» 2017 (взрослые): анемия — Hb < 130 г/л у мужчин и < 120 г/л
    у женщин; лёгкая 90–120, средняя 70–89, тяжёлая < 70 г/л. Только по Hb: это не диагноз ЖДА (статус
    железодефицита — отдельно, iron_deficiency_status). Отдельная система: с категориями ВОЗ не смешивается.
    → dict: status ok / insufficient / not_applicable / invalid; grade (None, если протокол степень не задаёт)."""
    out = {'system': 'kz2017', 'source': KZ2017_SOURCE, 'status': 'ok', 'grade': None, 'grade_ru': None,
           'anemic': None, 'normal_from_g_l': None, 'message': '', 'notes': []}
    try:
        hb = _norm_hb(hb_g_l)
        s, a, p, _ = _inputs(sex, age, pregnant, None)
    except _Invalid as e:
        return {**out, 'status': 'invalid', 'message': f'Противоречивые или неверные данные: {e}.'}
    if a is None:
        return {**out, 'status': 'insufficient', 'message': 'Недостаточно данных: укажите возраст.'}
    if a < KZ_MIN_AGE:
        return {**out, 'status': 'not_applicable', 'message': 'Не применяется: протокол — для взрослых.'}
    if p is True:
        return {**out, 'status': 'not_applicable',
                'message': 'Не применяется: пороги протокола даны без учёта беременности.'}
    if p is None and s != 'M':
        return {**out, 'status': 'insufficient',
                'message': 'Недостаточно данных: не указано, есть ли беременность (для беременных протокол '
                           'не применяется).'}
    cuts = [130] if s == 'M' else [120] if s == 'F' else [120, 130]
    if hb >= max(cuts):
        return {**out, 'grade': 'norm', 'grade_ru': KZ_GRADE_RU['norm'], 'anemic': False,
                'normal_from_g_l': float(max(cuts))}
    if hb >= min(cuts):     # пол не указан, Hb 120–129: у мужчин анемия, у женщин нет
        return {**out, 'status': 'insufficient',
                'message': f'Недостаточно данных: при Hb {_hb_text(hb)} г/л ответ зависит от пола '
                           '(мужчины: анемия, женщины: нет). Укажите пол.'}
    out.update(anemic=True, normal_from_g_l=float(max(cuts)))   # пол не указан: порог выше из двух, как у ВОЗ
    if hb > 120:             # мужчины 120–129: анемия, но шкала лёгкой степени заканчивается на 120
        return {**out, 'grade_ru': 'Анемия; степень шкалой протокола не задана (лёгкая — 90–120 г/л)'}
    grade = 'severe' if hb < 70 else 'moderate' if hb < 90 else 'mild'
    return {**out, 'grade': grade, 'grade_ru': KZ_GRADE_RU[grade]}


def iron_deficiency_status():
    """Статус железодефицита — отдельное поле, не степень анемии. Показатели обмена железа Я3 не принимает,
    поэтому подтверждения нет никогда; пороги ферритина и TSat здесь сознательно не применяются (вопрос врачу)."""
    return {'status': 'not_confirmed', 'confirmed': False, 'text': IRON_NOT_CONFIRMED}


def _system_line(name, c, normal_word):
    if c['status'] != 'ok':
        return f'{name}: {c["message"]}'
    cut = f'{normal_word} ≥ {c["normal_from_g_l"]:.0f} г/л' if c['normal_from_g_l'] else ''
    group = c.get('group')
    head = f'{name} ({", ".join(x for x in (group, cut) if x)})'
    return f'{head}: {c["grade_ru"].lower()}.' + ''.join(f' {n}' for n in c['notes'])


def lab_report(value, unit=None, sex=None, age=None, pregnant=None, trimester=None, elevation_m=None):
    """Интерпретация готового лабораторного Hb (ячейка Я3): ВОЗ 2024 и протокол МЗ РК 2017 — двумя отдельными
    системами. С оценкой по фото не связана: цифры фото-модели здесь не используются.
    → dict: прежние ключи (grade, grade_ru, anemic, normal_from_g_l, group — по ВОЗ 2024), who2024, kz2017,
    iron_deficiency (отдельный статус железодефицита), text. Этот же отчёт получает персонализатор (Я5),
    чтобы нормы в Я3 и Я5 совпадали. elevation_m — высота проживания, м (поправка ВОЗ 2024 только к
    шкале ВОЗ и только если высота указана; шкала МЗ РК без поправки).
    Ошибка числа или единицы → ValueError. Если по ВОЗ 2024 классифицировать нельзя (данных недостаточно или
    противоречие) → LabDataError (подкласс ValueError) с понятным сообщением и полным отчётом в .report."""
    hb = hb_to_g_l(value, unit)
    who = classify_who2024(hb, sex, age, pregnant, trimester, elevation_m)
    kz = classify_kz2017(hb, sex, age, pregnant)
    iron = iron_deficiency_status()
    adjust = ('Поправка на высоту применена только к ВОЗ 2024; в шкале МЗ РК её нет. ' if who['altitude'] else
              'Высота проживания не указана: поправка на высоту не применена. ') + NO_SMOKING_ADJUSTMENT
    text = '\n'.join([
        f'Hb {_hb_text(hb)} г/л (лабораторный анализ).',
        '• ' + _system_line('ВОЗ 2024', who, 'норма'),
        '• ' + _system_line('МЗ РК 2017, степень анемии по шкале протокола, взрослые', kz, 'норма'),
        '• ' + iron['text'],
        'ВОЗ и МЗ РК — две отдельные системы: их категории и пороги не совпадают. Степень анемии — не диагноз ЖДА. '
        + adjust,
        NOT_DIAGNOSIS])
    report = {'source': 'лабораторный анализ', 'hb_g_l': round(hb, 1), 'status': who['status'],
              'grade': who['grade'], 'grade_ru': who['grade_ru'], 'anemic': who['anemic'],
              'normal_from_g_l': who['normal_from_g_l'], 'group': who['group'],
              'who2024': who, 'kz2017': kz, 'iron_deficiency': iron, 'text': text}
    if who['status'] != 'ok':
        raise LabDataError(who['message'], report)
    return report


def lab_hb_report(value, unit=None, sex=None, age=None, pregnant=None):
    """Интерпретация готового лабораторного Hb по нормам ВОЗ 2011. Никак не связана с оценкой по фото:
    это отдельный путь, и цифры фото-модели здесь не используются. → dict.
    Прежняя версия без изменений: ею пользуются интерфейс HemaVision и персонализатор. Ячейка Я3 — lab_report
    (ВОЗ 2024 + протокол МЗ РК 2017)."""
    hb = hb_to_g_l(value, unit)
    lim = who_limits(sex, age, pregnant)
    grade = who_grade(hb, sex, age, pregnant)
    who = ('беременные' if is_pregnant(pregnant) else 'дети до 5 лет' if age is not None and age < 5 else
           'дети 5–11 лет' if age is not None and age < 12 else 'дети 12–14 лет' if age is not None and age < 15 else
           'мужчины ≥ 15 лет' if sex == 'M' else 'женщины ≥ 15 лет' if sex == 'F' else 'пол не указан: нормы женщин')
    return {'source': 'лабораторный анализ', 'hb_g_l': round(hb, 1), 'grade': grade, 'grade_ru': GRADE_RU[grade],
            'anemic': grade != 'norm', 'normal_from_g_l': lim.normal, 'group': who,
            'text': f'Hb {hb:.0f} г/л — {GRADE_RU[grade].lower()} (ВОЗ, {who}: норма ≥ {lim.normal:.0f} г/л). '
                    'Это интерпретация анализа крови, не оценка по фото; диагноз ставит врач.'}
