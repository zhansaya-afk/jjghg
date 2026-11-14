"""Опрос: 1 обязательный вопрос о красных флагах, 3 базовых и до 2 уточняющих.

Пол, возраст и беременность не спрашиваются (они уже есть до фото). Ответы меняют только объяснение
и список анализов; риск по фото они не меняют (данных, чтобы проверить такое влияние, нет).
Ответ: True / False / None («не знаю» или пропущен).
"""

from dataclasses import dataclass

from . import facts as F


@dataclass(frozen=True)
class Question:
    key: str
    ru: str
    kk: str
    required: bool = False


QUESTIONS = (
    Question(
        "red_flag",
        "За последние 2 недели было что-то из этого: обморок; боль в груди или одышка в покое; стул чёрный, как "
        "дёготь (если не принимаете железо); рвота с кровью; кровотечение при беременности?",
        "Соңғы 2 аптада мыналардың бірі болды ма: есінен тану; кеуде ауыруы немесе тыныштықта ентігу; қара, "
        "шайыр тәрізді нәжіс; қан аралас құсу; жүктілік кезінде қан кету?",
        required=True,
    ),
    Question(
        "weakness",
        "Бывают ли слабость, одышка или головокружение при обычной нагрузке?",
        "Әдеттегі жүктемеде әлсіздік, ентігу немесе бас айналу бола ма?",
    ),
    Question(
        "blood_loss",
        "Есть ли кровопотери: обильные месячные, донорство чаще раза в 3 месяца, кровь в стуле, частые носовые "
        "кровотечения?",
        "Қан жоғалту бар ма: етеккірдің көп болуы, 3 айда бір реттен жиі донорлық, нәжісте қан, мұрыннан жиі қан кету?",
    ),
    Question(
        "meat_fish",
        "Едите ли вы мясо или рыбу хотя бы 2–3 раза в неделю?",
        "Аптасына кемінде 2–3 рет ет немесе балық жейсіз бе?",
    ),
    Question(
        "family_thalassemia",
        "Есть ли у родственников талассемия или другая наследственная анемия?",
        "Туыстарыңызда талассемия немесе басқа тұқым қуалайтын анемия бар ма?",
    ),
    Question(
        "chronic_disease",
        "Есть ли хроническая болезнь почек или желудка и кишечника?",
        "Бүйректің немесе асқазан-ішектің созылмалы ауруы бар ма?",
    ),
)
BY_KEY = {q.key: q for q in QUESTIONS}


def _concern(facts):
    ph, lab = facts.get("photo") or {}, facts.get("lab") or {}
    return (
        ph.get("status") == "ok" and ph.get("risk") in ("средний", "высокий")
    ) or lab.get("grade") not in (
        None,
        "norm",
    )


def questions_for(facts):
    """Какие вопросы задать этому человеку, по порядку. Флаг — всегда первым; уточняющие — по ситуации."""
    if F.is_child(facts):
        return []
    keys = ["red_flag", "weakness", "blood_loss", "meat_fish"]
    if _concern(facts):
        keys.append("family_thalassemia")
        p = facts.get("profile") or {}
        older = p.get("age") is not None and p["age"] >= 50
        if p.get("sex") == "M" or older:
            keys.append("chronic_disease")
    return [BY_KEY[k] for k in keys]


def to_survey(answers, facts):
    """Ответы опроса {key: True/False/None} → блок survey карточки фактов.
    blood_loss у женщины трактуется и как heavy_periods (вопрос включает месячные)."""
    sv = {k: answers.get(k) for k in F.SURVEY_KEYS if k in answers}
    if answers.get("blood_loss") and (facts.get("profile") or {}).get("sex") == "F":
        sv.setdefault("heavy_periods", True)
    if answers.get("chronic_disease") is True:
        sv["chronic_disease"] = "почки или ЖКТ"
    return sv
