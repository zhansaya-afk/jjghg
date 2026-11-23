"""Карточка фактов персонализатора: результат фото, анализ крови, профиль и опрос → один словарь.

По фото ничего не пересчитывается: поля только переносятся из результата ядра. Фото и анализ крови —
два отдельных блока, друг на друга не влияют. Норма ВОЗ берётся из anemia.labs.
"""

import re

from ..labs import (
    LabDataError,
    classify_who2024,
    lab_report,
    who2024_normal,
    who_limits,
)

# Точность ядра, которую персонализатор называет пользователю. Источник: results/metrics_history.csv,
# task core, dataset eyes_defy, 2026-10-31 (f58eb53, calib_before): n = 193, AUC 0,856,
# чувствительность 0,837, специфичность 0,710. После нового evaluate.py обновлять только здесь.
MODEL_CARD = {
    "sensitivity_pct": 84,
    "specificity_pct": 71,
    "auc": 0.86,
    "patients": 193,
    "anemic": 86,  # tp + fn той же строки (72 + 14)
    "dataset": "Eyes-defy: взрослые, Италия и Индия, снимки с насадки",
}
PHOTO_KEYS = (
    "status",
    "stage",
    "risk",
    "p_anemia",
    "hb_interval_g_l",
    "hb_estimate_g_l",
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
    if out["status"] == "ok" and out["hb_interval_g_l"]:
        # число Hb по фото пользователю не показывается (predict.HB_USER): в карточку Personalizer цифры не попадают
        out.update(hb_interval_g_l=None, hb_estimate_g_l=None, hb_hidden=True)
    if out["status"] != "ok":
        out.update(
            risk="не определён",
            p_anemia=None,
            hb_interval_g_l=None,
            hb_estimate_g_l=None,
            tissue=None,
        )
    else:
        out["validation"] = dict(MODEL_CARD)
    return out


def lab_facts(lab, profile):
    """Отчёт labs.lab_report() (ячейка Я3, ВОЗ 2024), число г/л или строка с единицей → блок lab (или None).
    Число пересчитывается по ВОЗ 2024 для профиля (триместр — если указан). Если категорию определить
    нельзя (например, беременность без триместра при Hb 95–99 или 105–109 г/л) — grade None и message."""
    if lab is None:
        return None
    if (
        isinstance(lab, dict)
        and lab.get("grade") is None
        and lab.get("hb_g_l") is not None
    ):
        lab = lab["hb_g_l"]  # только число без степени → пересчитать по ВОЗ для профиля
    if not isinstance(lab, dict):
        try:
            lab = lab_report(
                lab,
                sex=profile.get("sex"),
                age=profile.get("age"),
                pregnant=profile.get("pregnant"),
                trimester=profile.get("trimester"),
            )
        except LabDataError as e:  # данных не хватает: степень не выдумываем
            lab = {**e.report, "message": str(e)}
    keys = ("hb_g_l", "grade", "grade_ru", "normal_from_g_l", "group", "message")
    return {k: lab.get(k) for k in keys}


def build_facts(result=None, lab=None, profile=None, survey=None):
    """Единственный источник правды о человеке для персонализатора."""
    profile = {"sex": None, "age": None, "pregnant": None, **(profile or {})}
    # триместр — только если указан (Я3); без него ключа нет, профиль как раньше
    profile = {k: v for k, v in profile.items() if k != "trimester" or v not in (None, "")}
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
    """Порог нормы ВОЗ 2024 для профиля (как в Я3). Возраст не указан — порог взрослых по полу (ВОЗ 2011 и 2024
    для взрослых совпадают), чтобы ответ не остался без числа."""
    p = facts.get("profile") or {}
    n = who2024_normal(p.get("sex"), p.get("age"), p.get("pregnant"), p.get("trimester"))
    if n is None:
        n = who_limits(p.get("sex"), p.get("age"), p.get("pregnant")).normal
    return round(n)


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


MMOL_G_L = 16.11  # 1 ммоль/л Hb (мономер) = 16,11 г/л, как в labs.py
_NUM = re.compile(r"(?<![\w.,])(\d{1,3})(?:[.,](\d))?(?!\d|[.,]\d)")
_HB_WORD = re.compile(
    r"гемогл|гемаглоб|\bhb\b|\bhgb\b|\bнв\b|гемоглобин", re.IGNORECASE
)
# гликированный гемоглобин (HbA1c, %) — показатель сахара, не Hb: «HbA1c 8» ≠ 80 г/л
GLYCATED = re.compile(r"гликир|гликозилир|hb\s*a1c|\ba1c\b|нв\s*а1с|\bа1с\b", re.IGNORECASE)
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
# казахские единицы целым словом: «10 жаста», «8 айлық», «2 апта», «3 күн», «2 рет», «70 пайыз», «5 жыл»;
# не «95 айналасында», «105 жылдам», «98 рет-рет» (там число — Hb)
_KK_OTHER_UNIT = re.compile(
    r"(?:жас(?:та|ар|қа|ка|ында|ымда|тағы|тагы)?|ай(?:лық|лык|да|дан|ға|га)?|апта\w*|к[үу]н(?:де|нен|ге)?|"
    r"рет(?!-)|пайыз\w*|жыл(?:да|дан|ға|га)?|са[ғг]ат\w*)\b"
)


def _last(rx, text):
    ms = list(rx.finditer(text))
    return ms[-1].start() if ms else -1


def hb_from_question(question, facts):
    """Число Hb из вопроса (если анализа в карточке нет): (г/л, единица в вопросе) или None.
    Единица: «г/л»; «г/дл» — число 3–25 (×10); «ммоль/л» — если так написано (×16,11, как labs.py).
    Возраст, недели, проценты и числа других показателей («ферритин 8», «MCV 75», «давление 120»)
    игнорируются."""
    if GLYCATED.search(question or ""):
        return None
    for m in _NUM.finditer(question or ""):
        whole, dec = int(m.group(1)), m.group(2)
        tail = question[m.end() : m.end() + 12].lower().lstrip(" ,.")
        mmol = tail.startswith(("ммоль", "mmol"))
        if (tail.startswith(_OTHER_UNIT) or _KK_OTHER_UNIT.match(tail)) and not mmol:
            continue
        head = question[max(0, m.start() - 30) : m.start()]
        if _last(_OTHER_WORD, head) > _last(
            _HB_WORD, head
        ):  # «ферритин 8», «MCV 75» — не гемоглобин
            continue
        v = whole + (int(dec) / 10 if dec else 0)
        if mmol and 2 <= v < 16:
            return round(v * MMOL_G_L), "ммоль/л"
        if 3 <= v < 25:
            return round(v * 10), "г/дл"
        if 30 <= v <= 250 and not dec:
            return round(v), "г/л"
    return None


def grade_of(hb, facts):
    """Категория ВОЗ 2024 для числа из вопроса; None — нельзя определить без недостающих данных."""
    p = facts.get("profile") or {}
    return classify_who2024(hb, p.get("sex"), p.get("age"), p.get("pregnant"), p.get("trimester"))["grade"]
