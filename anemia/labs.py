"""Нормы ВОЗ по гемоглобину. Все значения Hb в г/л.

Источник: WHO, Haemoglobin concentrations for the diagnosis of anaemia and assessment of severity (2011),
таблица 1. Порог «анемия» — ниже нормы; степени: лёгкая / умеренная / тяжёлая.
"""
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
