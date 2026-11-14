"""Память в пределах одного разговора: короткие уточнения и факты, сказанные в чате.

Ничего не сохраняется между запусками: всё берётся из списка history текущего разговора.
Факты из чата меняют только советы и список анализов (как ответы опроса); риск по фото, анализ и
профиль (пол, возраст, беременность) они не меняют.
"""

import re

from .facts import SURVEY_KEYS
from .retrieve import _norm
from .router import GREETING, THANKS

# Уточнение без своей темы: только вопросительные слова и «для кого» — «А для мужчин?», «А при
# беременности?», «А если 105?», «Почему?». «А если я донор?» — уже своя тема (донорство), ищется сама.
_ASK = """
    почему зачем сколько когда где какой какая какое какие каким чем куда откуда насколько долго надолго
    часто сильно тогда
"""
ASK_WORDS = frozenset(_ASK.split())
GROUP = re.compile(
    r"^(?:муж\w*|мужик\w*|парн\w*|жен\w*|девуш\w*|дет\w*|реб[её]н\w*|подрост\w*|пожил\w*|беремен\w*|"
    r"кормящ\w*|лактац\w*|\d+(?:[.,]\d+)?|г/л|г/дл|лет|год\w*|возраст\w*)$"
)
SHORT_MARK = re.compile(r"^\s*(?:ну\s+)?(?:а|и|тогда)\b", re.IGNORECASE)
# «А сколько?» после вопроса о железе / витаминах — это вопрос о дозе
AMOUNT = re.compile(
    r"^\W*(?:ну\s+)?(?:а|и)?\s*(?:по\s+)?сколько\b(?!\s+(?:ждать|времени|дней|недел\w*|месяц\w*|стоит|раз\s+сдавать))",
    re.IGNORECASE,
)
MED = re.compile(
    r"желез|препарат|таблет|витамин|капсул|лекарств|\bb12\b|фоли", re.IGNORECASE
)

# (поле опроса, значение, шаблон, шаблон сам содержит «не»). Только утверждения о себе: вопрос
# «что есть веганам?» не значит, что человек веган. Перед найденным «не / нет / ни» → не считается.
_SELF_KIDNEY = r"(?:болезн\w*\s+почек|почечн\w*\s+недостаточн\w*|пиелонефрит\w*|гломерулонефрит\w*)"
_SELF_GUT = r"(?:язв\w*|болезн\w*\s+крона|целиаки\w*|неспецифическ\w*\s+колит\w*|язвенн\w*\s+колит\w*)"
_KIN = r"(?:мам\w*|пап\w*|матер\w*|от[её]?ц\w*|семь\w*|родн\w*|родственник\w*|брат\w*|сестр\w*|бабушк\w*|дедушк\w*)"
_NO_NEG = r"(?:(?!(?:не|нет|ни)\b)\w+\s+){0,2}"
# fmt: off
DIALOG_FACTS = (
    ("meat_fish", False, r"\bя\s+(?:[-—]\s*)?(?:строг\w*\s+)?(?:веган|вегетариан)\w*", False),
    ("meat_fish", False, r"\b(?:стал|стала|перешл\w*|перешел|перешёл)\s+(?:на\s+)?(?:веган|вегетариан)\w*", False),
    ("meat_fish", False, r"\bне\s+(?:ем|ела|ел|едим|употребляю)\s+(?:ни\s+)?(?:мяс|рыб)", True),
    ("meat_fish", False, r"\b(?:мяс\w*|рыб\w*)(?:\s+и\s+(?:мяс|рыб)\w*)?\s+(?:я\s+)?(?:совсем\s+)?не\s+(?:ем|ела|ел)\b", True),
    ("meat_fish", True, r"\b(?:я\s+)?(?:ем|едим)\s+(?:(?!мало|редко)\w+\s+)?(?:мяс|рыб)", False),
    ("heavy_periods", True, rf"\bу\s+меня\s+{_NO_NEG}(?:обильн|сильн|длительн|долг)\w*\s+(?:месячн|менстру)", False),
    ("heavy_periods", True, r"\b(?:мои\s+)?месячн\w*\s+(?:у\s+меня\s+)?(?:очень\s+|слишком\s+)?(?:обильн|сильн)\w*", False),
    ("donor", True, r"\bя\s+(?:[-—]\s*)?(?:регулярн\w*\s+|почетн\w*\s+|почётн\w*\s+)?донор", False),
    ("donor", True, r"\b(?:являюсь|бываю|была|был|стала|стал)\s+донором", False),
    ("donor", True, r"\b(?:регулярно|часто|постоянно)\s+сда(?:ю|ём|ем|вала|вал)\s+(?:донорск\w*\s+)?кровь", False),
    ("donor", True, r"\bсда(?:ю|ём|ем)\s+(?:донорск\w*\s+кровь|кровь\s+(?:как\s+донор|регулярно|часто|кажд\w*))", False),
    ("family_thalassemia", True, rf"\b(?:у|в)\s+(?:меня|{_KIN})\s+(?:тоже\s+)?(?:есть\s+|была\s+|был\s+)?(?:бета[-\s]?)?талассеми", False),
    ("family_thalassemia", True, rf"\bталассеми\w*\s+(?:есть\s+)?(?:у|в)\s+(?:меня|{_KIN})", False),
    ("family_thalassemia", True, rf"\bнаследственн\w*\s+анеми\w*\s+(?:есть\s+)?(?:у|в)\s+(?:меня|{_KIN})", False),
    ("chronic_disease", "почки", rf"\bу\s+меня\s+{_NO_NEG}{_SELF_KIDNEY}", False),
    ("chronic_disease", "почки", r"\b(?:я\s+)?на\s+(?:гемо)?диализе", False),
    ("chronic_disease", "ЖКТ", rf"\bу\s+меня\s+{_NO_NEG}{_SELF_GUT}", False),
)
# fmt: on
_FACT_RX = tuple(
    (k, v, re.compile(p, re.IGNORECASE), neg) for k, v, p, neg in DIALOG_FACTS
)
_NEG = re.compile(
    r"\b(?:не|нет|ни|если)\b", re.IGNORECASE
)  # «а если я донор?» — не утверждение
SAID = {  # как подтвердить человеку, что факт учтён
    ("meat_fish", False): "вы не едите мясо или рыбу",
    ("meat_fish", True): "вы едите мясо или рыбу",
    ("heavy_periods", True): "у вас обильные месячные",
    ("donor", True): "вы сдаёте кровь как донор",
    ("family_thalassemia", True): "у вас или у родственников есть талассемия",
    ("chronic_disease", "почки"): "у вас болезнь почек",
    ("chronic_disease", "ЖКТ"): "у вас болезнь желудка или кишечника",
}


def user_texts(history):
    return [h.get("text", "") for h in (history or []) if h.get("role") == "user"]


def is_follow_up(question):
    """Уточнение прошлого вопроса без своей темы: «А для мужчин?», «А если 105?», «Почему?»."""
    if (
        not (question or "").strip()
        or GREETING.search(question)
        or THANKS.search(question)
    ):
        return False
    words = _norm(question).split()
    return len(words) <= 4 and all(w in ASK_WORDS or GROUP.match(w) for w in words)


def is_short_mark(question):
    """«А ферритин?», «А железо тоже?» — может быть и уточнением, и новым вопросом: сначала ищется сам."""
    return bool(SHORT_MARK.match(question or "")) and len(_norm(question).split()) <= 3


def base_question(history):
    """Последний самостоятельный вопрос пользователя (не уточнение, не приветствие) или ''."""
    for t in reversed(user_texts(history)):
        t = t.strip()
        if (
            t
            and not is_follow_up(t)
            and not GREETING.search(t)
            and not THANKS.search(t)
        ):
            return t
    return ""


def context_query(question, history, weak=False):
    """Уточнение → «последний самостоятельный вопрос + уточнение» (для маршрута, поиска и подстановки).
    weak=True — так же для «А ферритин?» (вызывается, только если сам вопрос не нашёлся)."""
    if not (is_follow_up(question) or (weak and is_short_mark(question))):
        return question
    base = base_question(history)
    return f"{base.rstrip('?.! ')}? {question.strip()}" if base else question


def asks_dose(question, query):
    """«А сколько?» после вопроса о железе или витаминах → вопрос о дозе."""
    return (
        query != question
        and bool(AMOUNT.match(question or ""))
        and bool(MED.search(query))
    )


def said_facts(texts):
    """Утверждения о себе в репликах → {поле опроса: значение}. Позднее сказанное заменяет раннее."""
    out = {}
    for t in texts:
        for key, value, rx, neg_ok in _FACT_RX:
            for m in rx.finditer(t or ""):
                before = t[max(0, m.start() - 15) : m.start()]
                if not neg_ok and (_NEG.search(before) or _NEG.search(m.group(0))):
                    continue
                out[key] = value
    return out


def dialog_facts(facts, history, question=""):
    """Карточка фактов + сказанное о себе в этом разговоре (вегетарианство, донорство, месячные…).
    Сказанное в чате новее ответов опроса и заменяет их. Исходная карточка не меняется."""
    said = said_facts([*user_texts(history), question or ""])
    if not said:
        return facts
    survey = {k: None for k in SURVEY_KEYS}
    survey.update(facts.get("survey") or {})
    survey.update(said)
    return {**facts, "survey": survey}


def new_facts_note(before, after):
    """Что нового учтено после реплики (для подтверждения в чате) или ''."""
    b, a = (before or {}).get("survey") or {}, (after or {}).get("survey") or {}
    new = [SAID[(k, v)] for k, v in a.items() if (k, v) in SAID and b.get(k) != v]
    return ("Учту до конца этого разговора: " + "; ".join(new) + ".") if new else ""
