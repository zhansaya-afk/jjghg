"""Карточка фактов персонализатора: результат фото, анализ крови, профиль и опрос → один словарь.

По фото ничего не пересчитывается: поля только переносятся из результата ядра. Фото и анализ крови —
два отдельных блока, друг на друга не влияют. Норма ВОЗ берётся из anemia.labs.
"""

import re

from ..labs import lab_hb_report, who_grade, who_limits

# Точность ядра, которую персонализатор называет пользователю. Источник: results/metrics_history.csv,
# task core, dataset eyes_defy, 2026-10-31 (f58eb53, calib_before): n = 193, AUC 0,856,
# чувствительность 0,837, специфичность 0,710. После нового evaluate.py обновлять только здесь.
MODEL_CARD = {
    "sensitivity_pct": 84,
    "specificity_pct": 71,
    "auc": 0.86,
    "patients": 193,
    "dataset": "Eyes-defy: взрослые, Италия и Индия, снимки с насадки",
}
PHOTO_KEYS = (
    "status",
    "stage",
    "risk",
    "p_anemia",
    "hb_interval_g_l",
    "tissue",
    "reason",
)
SURVEY_KEYS = (
    "red_flag",
    "weakness",
    "blood_loss",
    "heavy_periods",
    "meat_fish",
    "donor",
    "family_thalassemia",
    "chronic_disease",
)
_TISSUE = {("conj",): "конъюнктива", ("sclera",): "склера", ("conj", "sclera"): "обе"}


def _get(obj, key, default=None):
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def photo_facts(result):
    """api.Result, его to_dict() или dict из predict.analyze_photo → блок photo (или None)."""
    if result is None:
        return None
    out = {k: _get(result, k) for k in PHOTO_KEYS}
    if out["hb_interval_g_l"] is None and _get(
        result, "hb_range"
    ):  # сырой dict analyze_photo
        out["hb_interval_g_l"] = list(_get(result, "hb_range"))
    if out["tissue"] is None and _get(result, "tissues"):
        out["tissue"] = _TISSUE.get(tuple(_get(result, "tissues")))
    if out["reason"] is None and out["status"] != "ok" and _get(result, "issues"):
        # сырой dict analyze_photo: причина — первая проблема снимка, без совета после «—»
        out["reason"] = str(_get(result, "issues")[0]).split(" — ")[0].rstrip(".")
    if out["status"] != "ok":
        out.update(
            risk="не определён", p_anemia=None, hb_interval_g_l=None, tissue=None
        )
    else:
        out["validation"] = dict(MODEL_CARD)
    return out


def lab_facts(lab, profile):
    """Отчёт labs.lab_hb_report(), число г/л или строка с единицей → блок lab (или None)."""
    if lab is None:
        return None
    if (
        isinstance(lab, dict)
        and lab.get("grade") is None
        and lab.get("hb_g_l") is not None
    ):
        lab = lab["hb_g_l"]  # только число без степени → пересчитать по ВОЗ для профиля
    if not isinstance(lab, dict):
        lab = lab_hb_report(
            lab,
            sex=profile.get("sex"),
            age=profile.get("age"),
            pregnant=profile.get("pregnant"),
        )
    keys = ("hb_g_l", "grade", "grade_ru", "normal_from_g_l", "group")
    return {k: lab.get(k) for k in keys}


def build_facts(result=None, lab=None, profile=None, survey=None):
    """Единственный источник правды о человеке для персонализатора."""
    profile = {"sex": None, "age": None, "pregnant": None, **(profile or {})}
    sv = None
    if survey is not None:
        sv = {k: None for k in SURVEY_KEYS}
        sv.update({k: v for k, v in survey.items() if k in SURVEY_KEYS})
    return {
        "profile": profile,
        "photo": photo_facts(result),
        "lab": lab_facts(lab, profile),
        "survey": sv,
    }


# ------------------------------------------------------------------ производные факты (только из карточки)
def norm_g_l(facts):
    p = facts.get("profile") or {}
    return round(who_limits(p.get("sex"), p.get("age"), p.get("pregnant")).normal)


def group_name(facts):
    """Кому принадлежит норма: «беременных», «мужчин», «женщин» (по нормам ВОЗ из labs)."""
    p = facts.get("profile") or {}
    if p.get("pregnant") is True:
        return "беременных"
    if p.get("age") is not None and p["age"] < 15:
        return "детей этого возраста"
    return "мужчин" if p.get("sex") == "M" else "женщин"


def photo_ok(facts):
    ph = facts.get("photo") or {}
    return ph.get("status") == "ok"


def is_child(facts):
    p, ph = facts.get("profile") or {}, facts.get("photo") or {}
    return (p.get("age") is not None and p["age"] < 5) or ph.get("stage") == "age"


def is_pregnant(facts):
    return (facts.get("profile") or {}).get("pregnant") is True


def survey_flag(facts, key):
    return ((facts.get("survey") or {}).get(key)) is True


def survey_no(facts, key):
    return ((facts.get("survey") or {}).get(key)) is False


_NUM = re.compile(r"(?<![\w.,])(\d{1,3})(?:[.,](\d))?(?!\d|[.,]\d)")
_HB_WORD = re.compile(
    r"гемогл|гемаглоб|\bhb\b|\bhgb\b|\bнв\b|гемоглобин", re.IGNORECASE
)
_OTHER_WORD = re.compile(
    r"ферритин|\bmcv\b|\bmch|b12|в12|фоли|витамин|эритроцит|тромбоцит|лейкоцит|\bсоэ\b|сывороточн|\bттг\b|"
    r"давлен|пульс|температур|\bрост|\bвес|сахар|глюкоз|креатинин|ферритин|\bmch\b|rdw",
    re.IGNORECASE,
)
_OTHER_UNIT = (
    "недел",
    "лет",
    "год",
    "%",
    "мг",
    "мкг",
    "нг",
    "раз",
    "кг",
    "см",
    "мм",
    "фл",
    "мес",
    "дн",
    "час",
    "мин",
)


def _last(rx, text):
    ms = list(rx.finditer(text))
    return ms[-1].start() if ms else -1


def hb_from_question(question, facts):
    """Число Hb из вопроса (если анализа в карточке нет): (г/л, был ли это г/дл) или None.
    3–25 → г/дл (×10), 30–250 → г/л; возраст, недели, проценты и числа других показателей
    («ферритин 8», «MCV 75», «давление 120») игнорируются."""
    for m in _NUM.finditer(question or ""):
        whole, dec = int(m.group(1)), m.group(2)
        tail = question[m.end() : m.end() + 12].lower().lstrip(" ,.")
        if tail.startswith(_OTHER_UNIT):
            continue
        head = question[max(0, m.start() - 30) : m.start()]
        if _last(_OTHER_WORD, head) > _last(
            _HB_WORD, head
        ):  # «ферритин 8», «MCV 75» — не гемоглобин
            continue
        v = whole + (int(dec) / 10 if dec else 0)
        if 3 <= v < 25:
            return round(v * 10), True
        if 30 <= v <= 250 and not dec:
            return round(v), False
    return None


def grade_of(hb, facts):
    p = facts.get("profile") or {}
    return who_grade(hb, p.get("sex"), p.get("age"), p.get("pregnant"))
