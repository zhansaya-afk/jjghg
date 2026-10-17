"""Нормы ВОЗ по гемоглобину. Все значения Hb в г/л.

Источник: WHO, Haemoglobin concentrations for the diagnosis of anaemia and assessment of severity (2011),
таблица 1. Порог «анемия» — ниже нормы; степени: лёгкая / умеренная / тяжёлая.
"""
import re
from dataclasses import dataclass

GRADES = ('norm', 'mild', 'moderate', 'severe')
GRADE_RU = {'norm': 'Норма', 'mild': 'Лёгкая анемия', 'moderate': 'Умеренная анемия', 'severe': 'Тяжёлая анемия'}


@dataclass(frozen=True)
class WhoLimits:
    normal: float    # Hb ≥ normal → норма
    mild: float      # mild ≤ Hb < normal → лёгкая
    moderate: float  # moderate ≤ Hb < mild → умеренная; ниже → тяжёлая


def who_limits(sex=None, age=None, pregnant=False):
    """sex: 'M' / 'F' / None, age в годах (дробный для детей до 5 лет), pregnant: bool."""
    if pregnant or (age is not None and age < 5):
        return WhoLimits(110, 100, 70)
    if age is not None and age < 12:
        return WhoLimits(115, 110, 80)
    if age is not None and age < 15:
        return WhoLimits(120, 110, 80)
    if sex == 'M':
        return WhoLimits(130, 110, 80)
    return WhoLimits(120, 110, 80)


def who_grade(hb_g_l, sex=None, age=None, pregnant=False):
    lim = who_limits(sex, age, pregnant)
    if hb_g_l >= lim.normal:
        return 'norm'
    if hb_g_l >= lim.mild:
        return 'mild'
    if hb_g_l >= lim.moderate:
        return 'moderate'
    return 'severe'


def is_anemic(hb_g_l, sex=None, age=None, pregnant=False):
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


def lab_hb_report(value, unit=None, sex=None, age=None, pregnant=False):
    """Интерпретация готового лабораторного Hb по нормам ВОЗ 2011. Никак не связана с оценкой по фото:
    это отдельный путь, и цифры фото-модели здесь не используются. → dict."""
    hb = hb_to_g_l(value, unit)
    lim = who_limits(sex, age, pregnant)
    grade = who_grade(hb, sex, age, pregnant)
    who = ('беременные' if pregnant else 'дети до 5 лет' if age is not None and age < 5 else
           'дети 5–11 лет' if age is not None and age < 12 else 'дети 12–14 лет' if age is not None and age < 15 else
           'мужчины ≥ 15 лет' if sex == 'M' else 'женщины ≥ 15 лет' if sex == 'F' else 'пол не указан: нормы женщин')
    return {'source': 'лабораторный анализ', 'hb_g_l': round(hb, 1), 'grade': grade, 'grade_ru': GRADE_RU[grade],
            'anemic': grade != 'norm', 'normal_from_g_l': lim.normal, 'group': who,
            'text': f'Hb {hb:.0f} г/л — {GRADE_RU[grade].lower()} (ВОЗ, {who}: норма ≥ {lim.normal:.0f} г/л). '
                    'Это интерпретация анализа крови, не оценка по фото; диагноз ставит врач.'}
