"""Казахские слоты для базы знаний kb/kk.md — те же имена, что в texts.SLOTS, те же цифры из карточки фактов.

ЧЕРНОВИК ПЕРЕВОДА (ИИ, 2026-11-18): должен проверить носитель языка и врач. Цифры (нормы ВОЗ, анализ, точность
модели) берутся из тех же функций facts.py, что и в русских слотах, поэтому совпадают с русским ответом.
Причины непригодности снимка из analyze_photo приходят по-русски, поэтому в {photo_line} для плохого снимка —
общая казахская фраза «қайта түсіріңіз», без русской причины.
"""

import re

from . import facts as F
from .texts import RISK_KK, _gi_relevant, _has_hb, _hb, is_vegan, join, lab_grade

KK_GROUP_FOR = {  # «для кого» (для + родительный → «… үшін»)
    "беременных": "жүкті әйелдер",
    "мужчин": "ерлер",
    "женщин": "әйелдер",
    "детей этого возраста": "осы жастағы балалар",
}
GRADE_KK = {"mild": "жеңіл", "moderate": "орташа", "severe": "ауыр"}


def _group(f):
    return KK_GROUP_FOR.get(F.group_name(f), "әйелдер")


def doctor(f):
    """Кімге көрсету (барыс септік)."""
    return (
        "жүктілікті жүргізетін дәрігерге" if F.is_pregnant(f) else "терапевт дәрігерге"
    )


def photo_line(f):
    ph = f.get("photo")
    if not ph or F.is_child(f):
        return ""
    if ph.get("status") == "ok":
        return f"Фото бойынша анемия қаупі {RISK_KK.get(ph['risk'], ph['risk'])}."
    return "Бұл фотоны бағалау мүмкін емес: суретті қайта түсіріңіз."


def risk_meaning(f):
    if not F.photo_ok(f):
        return ""
    return {
        "высокий": "Жоғары қауіп фотодағы қабақ әдеттегіден бозарақ екенін білдіреді: осындай суреттері бар адамдарда "
        "гемоглобин жиі нормадан төмен болып шыққан.",
        "средний": "Орташа қауіп фото бойынша «нормаға ұқсайды» деп те, «анемияға ұқсайды» деп те сенімді айтуға "
        "болмайтынын білдіреді.",
        "низкий": "Төмен қауіп фотода бозару көрінбейтінін білдіреді, бірақ бұл анемияны жоққа шығармайды: әдіс қателесуі "
        "мүмкін, әсіресе жеңіл анемияда.",
    }.get(f["photo"].get("risk"), "")


def hb_photo(f):
    if not F.photo_ok(f):
        return ""
    # фото бойынша Hb саны пайдаланушыға көрсетілмейді (texts.hb_photo, predict.HB_USER сияқты)
    return "Бұл фото бойынша жүйе гемоглобин санын бермейді: мұндай баға сенімсіз болар еді."


def other_groups(f, question="", hb=None):
    """«Ерлер үшін ше?» — басқа топтың ДДҰ шегі (texts.other_groups сияқты)."""
    q, p = (question or "").lower(), f.get("profile") or {}
    out = []

    def add(text, thr):
        cmp = ""
        if hb is not None:
            cmp = f" {_hb(hb)} г/л " + (
                "бұл шектен төмен." if hb < thr else "бұл шектен төмен емес."
            )
        out.append(text + cmp)

    if (
        re.search(r"\bер(?:лер|кек|ге|дер)\w*|ұл\s+бала|жігіт", q)
        and p.get("sex") != "M"
    ):
        add("Ерлер үшін ДДҰ шегі жоғарырақ — 130 г/л.", 130)
        out.append(
            "Жауап профильдегі жынысыңыз бойынша берілді; ол қате болса, түзетіңіз."
        )
    elif re.search(r"әйел|кыз|қыз", q) and p.get("sex") != "F":
        add("Әйелдер үшін ДДҰ шегі — 120 г/л.", 120)
        out.append(
            "Жауап профильдегі жынысыңыз бойынша берілді; ол қате болса, түзетіңіз."
        )
    if re.search(r"жүкті|жукти", q) and not F.is_pregnant(f) and p.get("sex") != "M":
        add("Жүктілік кезінде ДДҰ 2024 шегі — I және III триместрде 110 г/л, II триместрде 105 г/л.", 110)
    return join(*out)


def norm_line(f, question=""):
    return join(
        f"{_group(f).capitalize()} үшін ДДҰ нормасы — {F.norm_g_l(f)} г/л және одан жоғары.",
        other_groups(f, question),
    )


def _lab_sentence(hb, grade, f, prefix=""):
    group, norm = _group(f), F.norm_g_l(f)
    if grade is None:  # ДДҰ 2024: санатты анықтау үшін деректер жетпейді (әдетте триместр); мәтінді қазақ тілін білетін адам тексеруі керек
        p = f.get("profile") or {}
        why = (
            "жүктілік кезінде шекті мән триместрге байланысты (I және III — 110 г/л, II — 105 г/л)"
            if p.get("pregnant") is True
            else "жыныс, жас немесе жүктілік толық көрсетілмеген"
        )
        return f"{prefix}{_hb(hb)} г/л — ДДҰ 2024 бойынша санатты анықтау үшін деректер жеткіліксіз: {why}."
    if grade == "norm":
        s = f"{prefix}{_hb(hb)} г/л — {group} үшін ДДҰ нормасы шегінде ({norm} г/л және одан жоғары)"
        sex = (f.get("profile") or {}).get("sex")
        if hb > (160 if sex == "F" else 170):
            return (
                s
                + ", бірақ әдеттегіден жоғары: жоғары гемоглобинді де дәрігерге көрсеткен жөн."
            )
        return s + (", бірақ төменгі шекке жақын." if hb < norm + 5 else ".")
    return (
        f"{prefix}{_hb(hb)} г/л — {group} үшін ДДҰ нормасынан ({norm} г/л) төмен, ДДҰ өлшемдері бойынша бұл "
        f"{GRADE_KK[grade]} дәреже."
    )


def lab_line(f, question=""):
    lab = f.get("lab")
    q = F.hb_from_question(question, f)
    if lab:
        mine = _lab_sentence(lab["hb_g_l"], lab["grade"], f, "Қан анализі: ")
        if q and abs(q[0] - lab["hb_g_l"]) >= 1:
            return join(
                _question_sentence(q, f, question),
                "Қосымшадағы анализіңіз: " + mine[len("Қан анализі: ") :],
            )
        return mine
    if q:
        return _question_sentence(q, f, question)
    return ""


def _question_sentence(q, f, question):
    hb, unit = q
    prefix = (
        "Егер бұл г/л-дегі гемоглобин болса, "
        if unit == "г/л"
        else f"Егер бұл {unit} болса, онда бұл "
    )
    s = _lab_sentence(hb, F.grade_of(hb, f), f, prefix) + (
        "" if unit == "г/л" else " Бланктегі бірлікті тексеріңіз."
    )
    return join(s, other_groups(f, question, hb))


def lab_advice(f, question=""):
    g = lab_grade(f, question)
    if g is None:
        if _has_hb(f, question):  # ДДҰ 2024: санат анықталмады — бәрібір дәрігерге
            return f"Анализді жақын күндері {doctor(f)} көрсетіңіз: ол жетіспейтін деректерді ескеріп бағалайды."
        return ""
    if g == "norm":
        return (
            "Егер бұл ретте әлсіздік немесе шаш түсу болса, дәрігерден ферритин туралы сұраңыз: гемоглобин қалыпты "
            "болса да, темір қоры төмен болуы мүмкін."
        )
    return (
        f"Себебі мен диагнозды дәрігер анықтайды: көбіне ферритинді (темір қорын) тексереді. Анализді жақын күндері "
        f"{doctor(f)} көрсетіңіз."
    )


def lab_vs_photo(f, question=""):
    g = lab_grade(f, question)
    if g is None or not F.photo_ok(f):
        return ""
    s = "Қан анализі фото бойынша бағадан дәлірек, сондықтан анализге сүйеніңіз: фото бойынша баға — тек скрининг, диагноз емес."
    risk, anemic = f["photo"]["risk"], g != "norm"
    if (risk == "высокий" and not anemic) or (risk == "низкий" and anemic):
        s += " Фото қателесуі мүмкін: жалған дабыл беруі (мысалы, жарыққа байланысты) немесе анемияны байқамай қалуы мүмкін."
    return s


def lab_missing(f, question=""):
    if F.photo_ok(f) and not f.get("lab") and not F.hb_from_question(question, f):
        return "Қосымшада қан анализі жоқ: егер оны тапсырсаңыз, гемоглобинді енгізіңіз — анализ фотодан дәлірек."
    return ""


def tissue_line(f):
    if not F.photo_ok(f):
        return ""
    t = f["photo"].get("tissue")
    if t == "склера":
        return (
            "Суретте төменгі қабақтың ішкі беті (конъюнктива) көрінбеді, сондықтан баға склера, яғни көздің ағы "
            "бойынша жасалды. Ол сенімділігі төменірек. Қайта түсірген дұрыс: қызғылт жолақ көрінуі үшін төменгі "
            "қабақты саусақпен төмен тартып, күндізгі жарықта түсіріңіз."
        )
    if t == "обе":
        return "Баға конъюнктива (төменгі қабақтың ішкі беті) мен склера бойынша бірге жасалды."
    return "Баға конъюнктива — төменгі қабақтың ішкі беті бойынша жасалды: оның түсі гемоглобинді ең жақсы көрсетеді."


def accuracy(f, question=""):
    v = ((f.get("photo") or {}).get("validation")) or F.MODEL_CARD
    q = (question or "").lower()
    note = ""
    if re.search(r"бала|жасөспір|жасоспир|жүкті|жукти", q):
        note = "Балалар, жасөспірімдер және жүкті әйелдер үшін дәлдік тексерілмеген: сондықтан қан анализіне сүйеніңіз."
    return join(
        f"Шынын айтсақ: {v['patients']} ересек адамға тексергенде жүйе анемиясы бар адамдардың {v['sensitivity_pct']}%-ын "
        f"тапты және анемиясы жоқ адамдардың {v['specificity_pct']}%-ын дұрыс белгіледі. Яғни ол екі жаққа да қателесуі "
        "мүмкін: анемияны байқамай қалуы немесе жалған дабыл беруі.",
        note,
        "Суреттер арнайы саптамамен түсірілген, кәдімгі телефонда дәлдік төменірек болуы мүмкін. Бұл скрининг, диагноз "
        "емес: нақты жауапты жалпы қан анализі береді.",
    )


def dataset_line(f):
    v = ((f.get("photo") or {}).get("validation")) or F.MODEL_CARD
    with_anemia = (
        f", олардың {v['anemic']}-ында ДДҰ өлшемдері бойынша анемия болған"
        if v.get("anemic")
        else ""
    )
    return (
        f"Жүйе {v['patients']} ересек адамда тексерілді{with_anemia}: ашық Eyes-defy жинағы (Италия мен Үндістан, "
        "қабақ суреттері арнайы саптамамен түсірілген, әркімнің гемоглобині өлшенген). Тексеру модель оқыту кезінде "
        f"көрмеген адамдарда жүрді. Анемиясы барлардың {v['sensitivity_pct']}%-ын тапты, анемиясы жоқтардың "
        f"{v['specificity_pct']}%-ын дұрыс белгіледі. Тексеруде балалар, жасөспірімдер, жүкті әйелдер және кәдімгі "
        "телефонмен түсірілген суреттер болған жоқ. Бұл скрининг, диагноз емес."
    )


def probability(f):
    if not F.photo_ok(f) or f["photo"].get("p_anemia") is None:
        return "Бұл сурет бойынша ықтималдық есептелмеген. Тек жалпы қан анализі арқылы тексеруге болады."
    p = f["photo"]["p_anemia"]
    return (
        f"{round(100 * p)}% ықтималдық — модельдің бағасы: түсі ұқсас суреті бар адамдардың шамамен 10-нан "
        f"{round(10 * p)}-ында гемоглобин нормадан төмен болып шыққан. Бұл сізде міндетті түрде анемия бар дегенді "
        "білдірмейді: бұл скрининг. Тек жалпы қан анализі арқылы тексеруге болады."
    )


def tests(f, question=""):
    items = ["MCV көрсеткіші бар жалпы қан анализі (ЖҚА)", "ферритин (темір қоры)"]
    if is_vegan(f, question) or re.search(
        r"вегетариан|веган|ет\s+жемей", (question or "").lower()
    ):
        items.append("B12 дәрумені мен фолий қышқылы")
    if F.survey_flag(f, "family_thalassemia"):
        items.append("гемоглобин электрофорезі")
    s = (
        "Әдетте мына анализдерден бастайды: "
        + ", ".join(items)
        + ". Түпкілікті тізімді дәрігер анықтайды."
    )
    if F.survey_flag(f, "heavy_periods") or re.search(
        r"етеккір|етеккир", (question or "").lower()
    ):
        s += " Етеккір көп болса, гинекологқа көрінген пайдалы."
    return s


def survey_hints(f, question=""):
    out = []
    q = (question or "").lower()
    if F.survey_flag(f, "heavy_periods") or re.search(r"етеккір|етеккир", q):
        out.append(
            "Әйелдерде жиі кездесетін себептердің бірі — етеккірдің көп болуы: онымен бірге темір жоғалады."
        )
    if is_vegan(f, question):
        out.append("Ет пен балық жемегенде темір мен B12 дәрумені жиі жетіспейді.")
    if F.survey_flag(f, "donor") or "донор" in q:
        out.append("Қанды жиі тапсырғанда темір қоры әрдайым қалпына келіп үлгермейді.")
    if F.survey_flag(f, "family_thalassemia"):
        out.append(
            "Талассемия тұқым қуалайды және гемоглобинді де төмендетеді, бірақ темір тапшылығы сияқты емделмейді, "
            "сондықтан темірді өз бетіңізше ішпеңіз."
        )
    cd = (f.get("survey") or {}).get("chronic_disease")
    if isinstance(cd, str) and "почк" in cd.lower():
        out.append(
            "Бүйрек эритроциттер түзілуіне қажет эритропоэтин гормонын шығарады; мұны нефрологпен талқылаңыз."
        )
    if _gi_relevant(f):
        out.append(
            "Ерлерде және етеккірі тоқтаған әйелдерде тамақтану қалыпты болса, жиі кездесетін мүмкін себеп — асқазан "
            "немесе ішектен жасырын қан кету, сондықтан дәрігерде тексерілу маңызды."
        )
    return join(*out)


def next_step(f):
    if f.get("lab"):
        return f"Анализді {doctor(f)} көрсетіңіз."
    if F.photo_ok(f):
        return f"Жақын күндері жалпы қан анализін тапсырып, нәтижесін {doctor(f)} көрсетіңіз."
    return ""


def test_need(f, question=""):
    concern = (F.photo_ok(f) and f["photo"]["risk"] in ("средний", "высокий")) or (
        f.get("lab") and f["lab"]["grade"] != "norm"
    )
    if concern:
        return join(
            photo_line(f),
            "Иә, тапсырған жөн: жақын күндері жалпы қан анализін тапсырыңыз.",
        )
    return (
        "Шұғыл емес: жалпы қан анализін жоспарлы тексеру кезінде тапсыруға болады. Фото бойынша төмен қауіп анемияны "
        "жоққа шығармайды: әдіс қателесуі мүмкін. Әлсіздік, ентігу, бас айналу немесе бозару пайда болса, жалпы қан "
        "анализін ертерек тапсырып, дәрігерге көрсетіңіз."
    )


SLOTS = {
    "photo_line": lambda f, q: photo_line(f),
    "risk_meaning": lambda f, q: risk_meaning(f),
    "hb_photo": lambda f, q: hb_photo(f),
    "norm_line": norm_line,
    "lab_line": lab_line,
    "lab_advice": lab_advice,
    "lab_vs_photo": lab_vs_photo,
    "lab_missing": lab_missing,
    "not_diagnosis": lambda f, q: (
        "Бұл скрининг, диагноз емес."
        if f.get("photo")
        else "Диагноз бен себебін тек дәрігер қояды."
    ),
    "exact_note": lambda f, q: (
        "Нақты мәнді тек қан анализі көрсетеді."
        if not f.get("lab")
        else "Қан анализі фотодан дәлірек."
        if F.photo_ok(f)
        else ""
    ),
    "pregnancy_iron": lambda f, q: (
        "Жүктілік кезінде темір мен фолий қышқылын алдын алу үшін де жиі тағайындайды, бірақ мұны жүктілікті "
        "жүргізетін дәрігер шешеді."
        if F.is_pregnant(f) or re.search(r"жүкті|жукти", (q or "").lower())
        else ""
    ),
    "tissue_line": lambda f, q: tissue_line(f),
    "accuracy": accuracy,
    "dataset_line": lambda f, q: dataset_line(f),
    "probability": lambda f, q: probability(f),
    "tests": tests,
    "survey_hints": survey_hints,
    "doctor": lambda f, q: doctor(f),
    "next_step": lambda f, q: next_step(f),
    "test_need": test_need,
}

DONT_KNOW = (
    "Бұл сұраққа менің базамда нақты жауап жоқ, білмеймін. Терапевт дәрігерден сұраңыз."
)
RETAKE_NOTE = (
    "Бұл фотоны бағалау мүмкін емес: нәтиже керек болса, суретті қайта түсіріңіз."
)
