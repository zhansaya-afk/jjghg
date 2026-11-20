"""PDF-отчёт HemaVision для пациента и врача (A4, офлайн, reportlab). Только отображение: без новых правил.

Источник данных — результат текущей проверки: r (predict.analyze_photo), res (api.to_result), анкета (view.Survey),
готовый анализ крови (labs.lab_hb_report, как в интерфейсе), ответы AI Personalizer этой сессии. Ничего не
пересчитывается: уровень риска, Hb, тексты причин и пояснений — из ядра; нормы — те же функции labs, что использует
ядро. Отсутствующие данные печатаются как «нет данных» / «не указано» / «не оценивался», демонстрационные значения
не подставляются никогда.
Правила как на экране результата: status ≠ 'ok' → ни уровня, ни Hb; Hb — только если ядро выдало hb_range, с
неопределённостью из ядра, пометкой «НЕ анализ крови» и оговоркой о точности (predict.HB_DOCTOR_CAVEAT); анализ крови — отдельным разделом.

Кириллица: нужен TTF-шрифт (DejaVu Sans / Liberation Sans / Arial). Ищется в системе и в matplotlib; путь можно
задать переменной HEMAVISION_FONT (обычный) и HEMAVISION_FONT_BOLD. Не найден → FontNotFound (без «квадратиков»).
"""
import datetime as dt
import io
import os
import re
from xml.sax.saxutils import escape

import numpy as np

from anemia import labs as LABS
from anemia import predict as PR

DISCLAIMER = 'Не является диагнозом. Не заменяет анализ крови и консультацию врача.'
TITLE = 'HemaVision — протокол скринингового исследования риска анемии по фото глаза'
NO_DATA = 'нет данных'
WHO2011_SOURCE = ('ВОЗ 2011: WHO, Haemoglobin concentrations for the diagnosis of anaemia and assessment of '
                  'severity (WHO/NMH/NHD/MNM/11.1), таблица 1')
LEVELS = ('низкий', 'средний', 'высокий')
Q_TEXT = {'ok': 'в норме', None: 'не проверялось'}
RU3 = {True: 'да', False: 'нет', None: 'не указано'}
FONT_CANDIDATES = (     # (обычный, жирный)
    ('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
    ('/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf',
     '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'),
    ('/usr/share/fonts/dejavu/DejaVuSans.ttf', '/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf'),
    ('/Library/Fonts/Arial Unicode.ttf', '/Library/Fonts/Arial Unicode.ttf'),
    ('C:/Windows/Fonts/arial.ttf', 'C:/Windows/Fonts/arialbd.ttf'),
)


class FontNotFound(RuntimeError):
    """Нет TTF-шрифта с кириллицей: PDF не формируется (иначе вместо букв были бы пустые квадраты)."""


def find_fonts():
    """→ (путь обычного, путь жирного) TTF с кириллицей. Жирного нет → обычный."""
    env = os.environ.get('HEMAVISION_FONT')
    cands = [(env, os.environ.get('HEMAVISION_FONT_BOLD') or env)] if env else []
    try:                                        # matplotlib (есть в Colab) везёт DejaVu Sans в mpl-data
        import matplotlib
        d = os.path.join(os.path.dirname(matplotlib.__file__), 'mpl-data', 'fonts', 'ttf')
        cands.append((os.path.join(d, 'DejaVuSans.ttf'), os.path.join(d, 'DejaVuSans-Bold.ttf')))
    except ImportError:
        pass
    for reg, bold in cands + list(FONT_CANDIDATES):
        if reg and os.path.isfile(reg):
            return reg, bold if bold and os.path.isfile(bold) else reg
    raise FontNotFound('не найден TTF-шрифт с кириллицей (DejaVu Sans / Liberation Sans): установите fonts-dejavu '
                       'или задайте путь в HEMAVISION_FONT')


_FONTS = {}


def _register_fonts():
    if not _FONTS:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
        reg, bold = find_fonts()
        pdfmetrics.registerFont(TTFont('HV', reg))
        pdfmetrics.registerFont(TTFont('HV-Bold', bold))
        pdfmetrics.registerFontFamily('HV', normal='HV', bold='HV-Bold', italic='HV', boldItalic='HV-Bold')
        _FONTS.update(regular=reg, bold=bold)
    return _FONTS


# ------------------------------------------------------------------ данные отчёта (чистые функции, без reportlab)
def _cap(s):
    s = str(s)
    return s[:1].upper() + s[1:]


def _plain(md):
    """Markdown ответа Personalizer → обычный текст (жирный ** убирается, списки остаются строками)."""
    return re.sub(r'\*\*(.+?)\*\*', r'\1', str(md or '')).strip()


def who2011_group(sex=None, age=None, pregnant=None):
    """Группа нормы ВОЗ 2011 словами — по тем же ветвям, что labs.who_limits (её и применяет ядро)."""
    if LABS.is_pregnant(pregnant):
        return 'беременные'
    if age is not None and age < 5:
        return 'дети до 5 лет'
    if age is not None and age < 12:
        return 'дети 5–11 лет'
    if age is not None and age < 15:
        return 'дети 12–14 лет'
    if sex == 'M':
        return 'мужчины ≥ 15 лет'
    if sex == 'F':
        return 'женщины ≥ 15 лет'
    return 'пол не указан: применены нормы женщин'


def references(r, survey, lab=None):
    """Применённые референсные значения: [(что, значение, источник)]. Только то, что реально применялось."""
    out = []
    if r.get('status') == 'ok':
        lim = LABS.who_limits(survey.sex, survey.age, survey.pregnant)
        out.append(('Оценка по фото (сверка диапазона Hb с нормой)',
                    f'норма Hb ≥ {lim.normal:.0f} г/л; {who2011_group(survey.sex, survey.age, survey.pregnant)}',
                    WHO2011_SOURCE))
    if lab:
        out.append(('Анализ крови (введён пользователем)',
                    f'норма Hb ≥ {lab["normal_from_g_l"]:.0f} г/л; {lab["group"]}', WHO2011_SOURCE))
    return out


def risk_block(r):
    """→ dict: level (уровень | None), title, text, issues. Без оценки — без уровня."""
    if r.get('status') == 'ok' and r.get('risk') in LEVELS:
        lv = r['risk']
        return {'level': lv, 'title': f'{lv.upper()} РИСК', 'text': _cap(PR.RISK_TEXT[lv]) + '.', 'issues': []}
    st = PR.STATUS_RU.get(r.get('status'), 'оценка не выполнена')
    return {'level': None, 'title': 'РИСК НЕ ОПРЕДЕЛЁН',
            'text': f'{_cap(st)}. Уровень риска и гемоглобин по фото не выдаются.',
            'issues': [str(x) for x in r.get('issues') or []]}


def hb_block(r):
    """→ (заголовок, текст). Hb только из hb_range ядра, с неопределённостью из hb_note."""
    if r.get('status') != 'ok':
        return 'Hb по фото: не оценивался', 'Оценка по фото не выполнена.'
    if r.get('hb_range'):
        note = r.get('hb_note') or 'неопределённость модели не указана'
        return (f'Hb по фото: {PR.hb_text(r)}',
                f'Это НЕ лабораторное измерение, а грубая оценка по цвету на фото. Неопределённость: {note}. '
                f'Внимание: {PR.HB_DOCTOR_CAVEAT}.')
    return 'Hb по фото: не показан', _cap(r.get('hb_note') or NO_DATA) + '.'


def quality_rows(res):
    """Качество снимка (api.Result.quality) → [(показатель, значение)]. res None → одна строка «не проверялось»."""
    if res is None:
        return [('Проверка качества', 'не выполнялась')]
    q = res.quality or {}
    vis = q.get('tissue_visible') or {}

    def tv(v):
        return 'не проверялось' if v is None else 'найдена' if v else 'не найдена'
    rows = [('Резкость', Q_TEXT.get(q.get('sharpness'), q.get('sharpness'))),
            ('Свет', Q_TEXT.get(q.get('light'), q.get('light'))),
            ('Слизистая нижнего века (конъюнктива)', tv(vis.get('conjunctiva'))),
            ('Белок глаза (склера)', tv(vis.get('sclera')))]
    rows.append(('Оценено по ткани', res.tissue or 'не оценивалось'))
    return rows


def survey_rows(survey):
    from .personalizer import WELLBEING
    rows = [('Пол', {'F': 'женский', 'M': 'мужской'}.get(survey.sex, 'не указан')),
            ('Возраст', 'не указан' if survey.age is None else f'{survey.age:g} лет')]
    if survey.sex == 'F':
        rows.append(('Беременность', RU3[survey.pregnant]))
    rows.append(('Ткань для оценки', PR.METHOD_RU.get(survey.method, 'Авто')))
    rows += [(q, RU3[survey.wellbeing.get(k)]) for k, q in WELLBEING]
    return rows


def personalizer_entries(reply=None, history=()):
    """Ответы Personalizer этой проверки → [(кто, текст)]. Недоступный ответ (заглушка, ошибка) не печатается."""
    out = []
    if reply is not None and getattr(reply, 'available', False):
        out.append((reply.title or 'AI Personalizer', _plain(reply.text)))
    for m in history or ():
        if m.get('intro'):
            continue                            # первое объяснение уже выше (reply)
        out.append(('Вопрос' if m.get('role') == 'user' else 'Ответ', _plain(m.get('text'))))
    return out


def report_id(created):
    return f'HV-{created:%Y%m%d-%H%M%S}'


# ------------------------------------------------------------------ изображения
def _photo_image(photo, max_side=1000):
    """Исходное фото (путь / bytes / ndarray RGB) → PIL RGB (уменьшенное) | None."""
    if photo is None:
        return None
    from PIL import Image, ImageOps
    try:
        if isinstance(photo, np.ndarray):
            a = photo if photo.dtype == np.uint8 else (np.clip(photo, 0, 1) * 255).astype(np.uint8)
            im = Image.fromarray(a[..., :3] if a.ndim == 3 else a)
        else:
            im = Image.open(io.BytesIO(photo) if isinstance(photo, (bytes, bytearray)) else photo)
            im = ImageOps.exif_transpose(im)
        im = im.convert('RGB')
    except Exception:   # noqa: BLE001 — нечитаемое фото: в отчёте «нет изображения», а не падение
        return None
    im.thumbnail((max_side, max_side))
    return im


def _np_image(a):
    if a is None:
        return None
    from PIL import Image
    return Image.fromarray(np.asarray(a, dtype=np.uint8))


def images(r, photo=None):
    """→ [(подпись, PIL)] : исходное фото, вырезка глаза с контурами, всё лицо с рамками (если было фото лица)."""
    out = []
    im = _photo_image(photo)
    if im is not None:
        out.append(('Исходное фото', im))
    for title, fn in (('Найденные ткани (маска сегментации)', PR.overlay), ('Глаза на фото', PR.overlay_photo)):
        try:
            a = fn(r) if (r.get('crop') is not None or r.get('photo_preview') is not None) else None
        except Exception:   # noqa: BLE001 — неполный результат: картинки нет
            a = None
        if a is not None:
            out.append((title, _np_image(a)))
    return out


# ------------------------------------------------------------------ сборка PDF (формат протокола исследования)
METHOD_TEXT = ('неинвазивная оценка по фотографии глаза: цветовой анализ слизистой нижнего века (конъюнктивы) '
               'и белка глаза (склеры), модель машинного обучения')
LIMITS_TEXT = ('Показатели модели получены ретроспективной кросс-валидацией на одном наборе данных (Eyes-defy: '
               'взрослые Италии и Индии, снимки через насадку-линзу). Клиническая валидация не проводилась; точность '
               'на других телефонах, при другом освещении и у жителей Казахстана не проверена. Программа не '
               'зарегистрирована как медицинское изделие. Детям младше 5 лет оценка не проводится.')
TISSUE_TEXT = {'обе': 'конъюнктива и склера', 'конъюнктива': 'конъюнктива', 'склера': 'склера'}


def result_rows(r, survey, lab=None):
    """Таблица результатов как в лабораторном бланке: [(показатель, результат, ед., референс, примечание)].
    Отсутствующее значение — словами, без единиц и норм (не выглядит как измерение)."""
    ok = r.get('status') == 'ok'
    rb = risk_block(r)
    risk = (rb['level'].upper() if rb['level'] else 'НЕ ОПРЕДЕЛЁН')
    rows = [('Риск анемии по фотографии', risk, '—', 'низкий / средний / высокий',
             'скрининговая оценка, не диагноз' if ok else 'оценка не выполнена')]
    if ok and r.get('hb_range'):
        lo, hi = r['hb_range']
        lim = LABS.who_limits(survey.sex, survey.age, survey.pregnant)
        rows.append(('Гемоглобин, оценка по фото', f'~{PR.hb_estimate_shown(r):.0f} (точно не определить: '
                     f'{lo:.0f}–{hi:.0f})', 'г/л', f'≥ {lim.normal:.0f}',
                     'НЕ анализ крови; неопределённость: ' + (r.get('hb_note') or 'не указана')
                     + '. Внимание: ' + PR.HB_DOCTOR_CAVEAT))
    elif ok:
        rows.append(('Гемоглобин, оценка по фото', 'не показан', '—', '—',
                     _cap(r.get('hb_note') or NO_DATA)))
    else:
        rows.append(('Гемоглобин, оценка по фото', 'не оценивался', '—', '—', 'оценка по фото не выполнена'))
    if lab:
        rows.append(('Гемоглобин, лабораторный (введён пациентом)', f'{lab["hb_g_l"]:.0f}', 'г/л',
                     f'≥ {lab["normal_from_g_l"]:.0f}', f'{lab["grade_ru"]}; документ анализа не проверялся'))
    else:
        rows.append(('Гемоглобин, лабораторный (введён пациентом)', 'не представлен', '—', '—', '—'))
    return rows


def build_pdf(path, r, survey, *, res=None, lab=None, photo=None, model_version='', reply=None, history=(),
              created=None):
    """Записать PDF-протокол в path → path. r — dict analyze_photo (полный, частичный или отказ), survey —
    view.Survey, res — api.Result (качество) или None, lab — labs.lab_hb_report или None, photo — исходное фото
    или None, reply / history — ответы Personalizer этой проверки (или None / ()).
    Формат — протокол исследования: паспортная часть, сведения об исследовании, таблица результатов с
    референсными значениями, контроль качества, иллюстрации, интерпретация, источники, ограничения, отметки врача,
    приложение с ответами Personalizer; чёрно-белый, читается при печати в оттенках серого."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas as rl_canvas
    from reportlab.platypus import (
        Image,
        KeepTogether,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    _register_fonts()
    created = created or dt.datetime.now()  # noqa: DTZ005 — местное время устройства, как в .txt-отчёте
    rid = report_id(created)
    version = model_version or 'модель не указана'
    ink, muted, grid, head_bg = (colors.HexColor(c) for c in ('#111111', '#444444', '#555555', '#e6e6e6'))
    st = {
        'org': ParagraphStyle('org', fontName='HV', fontSize=8.5, leading=11, textColor=muted),
        'h1': ParagraphStyle('h1', fontName='HV-Bold', fontSize=13, leading=17, textColor=ink, alignment=TA_CENTER),
        'sub': ParagraphStyle('sub', fontName='HV', fontSize=10, leading=13, textColor=ink, alignment=TA_CENTER),
        'h2': ParagraphStyle('h2', fontName='HV-Bold', fontSize=10.5, leading=14, textColor=ink, spaceBefore=9,
                             spaceAfter=3),
        'p': ParagraphStyle('p', fontName='HV', fontSize=9.5, leading=12.5, textColor=ink),
        'cell': ParagraphStyle('cell', fontName='HV', fontSize=9, leading=11.5, textColor=ink),
        'th': ParagraphStyle('th', fontName='HV-Bold', fontSize=8.5, leading=11, textColor=ink),
        'small': ParagraphStyle('small', fontName='HV', fontSize=8.5, leading=11, textColor=muted),
        'cap': ParagraphStyle('cap', fontName='HV', fontSize=8.5, leading=11, textColor=muted, alignment=TA_CENTER),
    }

    def P(text, style='p'):
        return Paragraph(escape(str(text)).replace('\n', '<br/>'), st[style])

    def table(rows, widths, head=None, bold_col=None):
        data = [[Paragraph(escape(h), st['th']) for h in head]] if head else []
        for row in rows:
            data.append([Paragraph(('<b>%s</b>' if j == bold_col else '%s') % escape(str(c)), st['cell'])
                         for j, c in enumerate(row)])
        t = Table(data, colWidths=widths, repeatRows=1 if head else 0)
        t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, grid), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                               ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                               ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
                              + ([('BACKGROUND', (0, 0), (-1, 0), head_bg)] if head else [])))
        return t

    def kv(rows):
        return table(rows, [W * 0.36, W * 0.64])

    W = A4[0] - 36 * mm
    ok = r.get('status') == 'ok'
    sex_ru = {'F': 'женский', 'M': 'мужской'}.get(survey.sex, 'не указан')
    age_ru = 'не указан' if survey.age is None else f'{survey.age:g}'

    # шапка
    top = Table([[P('HemaVision (PoC) — программа скрининга риска анемии. Не является медицинским изделием.', 'org'),
                  Paragraph(f'Протокол № {escape(rid)}<br/>от {created:%d.%m.%Y}', st['org'])]],
                colWidths=[W * 0.68, W * 0.32])
    top.setStyle(TableStyle([('ALIGN', (1, 0), (1, 0), 'RIGHT'), ('LINEBELOW', (0, 0), (-1, 0), 1, ink),
                             ('LEFTPADDING', (0, 0), (-1, -1), 0), ('RIGHTPADDING', (0, 0), (-1, -1), 0)]))
    story = [top, Spacer(1, 8), P('ПРОТОКОЛ СКРИНИНГОВОГО ИССЛЕДОВАНИЯ', 'h1'),
             P('Оценка риска анемии по фотографии глаза (неинвазивно)', 'sub'), Spacer(1, 6)]
    disc = Table([[Paragraph(f'<b>{escape(DISCLAIMER)}</b> Результат — скрининговая оценка риска по цвету тканей '
                             'глаза; наличие анемии устанавливается только по общему анализу крови.', st['p'])]],
                 colWidths=[W])
    disc.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 1.2, ink), ('LEFTPADDING', (0, 0), (-1, -1), 6),
                              ('TOPPADDING', (0, 0), (-1, -1), 5), ('BOTTOMPADDING', (0, 0), (-1, -1), 6)]))
    story.append(disc)

    # 1. Пациент
    from .personalizer import WELLBEING
    pat = [('Ф. И. О.', '_' * 46), ('Дата рождения', '_' * 20), ('Пол', sex_ru), ('Возраст, полных лет', age_ru)]
    if survey.sex == 'F':
        pat.append(('Беременность', RU3[survey.pregnant]))
    pat += [(q, RU3[survey.wellbeing.get(k)]) for k, q in WELLBEING]
    story += [P('1. Сведения о пациенте', 'h2'), kv(pat),
              P('Ф. И. О. и дату рождения заполняет пациент или медицинский работник: приложение их не хранит. '
                'Ответы о самочувствии — со слов пациента, на оценку риска не влияют.', 'small')]

    # 2. Исследование
    tissue = TISSUE_TEXT.get(res.tissue, res.tissue) if res is not None and res.tissue else 'не определена'
    story += [P('2. Сведения об исследовании', 'h2'),
              kv([('Дата и время', f'{created:%d.%m.%Y %H:%M}'), ('Метод', METHOD_TEXT),
                  ('Анализируемая ткань', tissue),
                  ('Выбор ткани', PR.METHOD_RU.get(survey.method, 'автоматически')),
                  ('Модель', r.get('model') or NO_DATA), ('Версия программы / модели', version),
                  ('Статус исследования', f'{PR.STATUS_RU.get(r.get("status"), r.get("status") or NO_DATA)} '
                                          f'(этап: {r.get("stage") or NO_DATA})')])]

    # 3. Результаты
    story += [P('3. Результаты', 'h2'),
              table(result_rows(r, survey, lab), [W * 0.23, W * 0.17, W * 0.07, W * 0.13, W * 0.40],
                    head=('Показатель', 'Результат', 'Ед.', 'Референс', 'Примечание'), bold_col=1),
              P('Референс гемоглобина — нижняя граница нормы ВОЗ 2011 для группы пациента (раздел 7).', 'small')]

    # 4. Качество
    story += [P('4. Контроль качества изображения', 'h2'), kv(quality_rows(res))]

    # 5. Иллюстрации
    story.append(P('5. Иллюстрации', 'h2'))
    ims = images(r, photo)
    if ims:
        cw = W / len(ims)
        row_im, row_cap = [], []
        for n, (cap, im) in enumerate(ims, 1):
            w = min(cw - 6, 62 * mm)
            h = w * im.height / im.width
            if h > 50 * mm:
                h, w = 50 * mm, 50 * mm * im.width / im.height
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=85)
            buf.seek(0)
            row_im.append(Image(buf, width=w, height=h))
            row_cap.append(Paragraph(f'Рис. {n}. {escape(cap)}', st['cap']))
        gal = Table([row_im, row_cap], colWidths=[cw] * len(ims))
        gal.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        legend = []
        if any(c.startswith('Найденные') for c, _ in ims):
            legend.append('Контуры на вырезе: зелёный — слизистая нижнего века, голубой — склера; в расчёт вошли '
                          'только эти области.')
        if any(c == 'Глаза на фото' for c, _ in ims):
            legend.append('Рамки на фото лица: зелёная — глаз принят, красная — не подошёл.')
        story.append(KeepTogether([gal] + [P(x, 'small') for x in legend]))
    else:
        story.append(P('Изображения недоступны: ' + NO_DATA + '.'))

    # 6. Интерпретация
    rb = risk_block(r)
    interp = [P(rb['text'])] + [P('• ' + x) for x in rb['issues']]
    if ok and r.get('reasons'):
        interp.append(P('Обоснование (автоматически): ' + '; '.join(str(x) for x in r['reasons']) + '.', 'small'))
    interp += [P('Примечание: ' + str(x)) for x in r.get('notes') or []]
    interp.append(P('Рекомендация: общий анализ крови с определением гемоглобина; оценка результата врачом. '
                    'Точно уровень гемоглобина показывает только анализ крови.'))
    story += [P('6. Интерпретация (формируется автоматически, не является заключением врача)', 'h2')] + interp

    # 7. Нормативные источники
    story.append(P('7. Применённые референсные значения и источники', 'h2'))
    refs = references(r, survey, lab)
    if refs:
        story.append(table(refs, [W * 0.3, W * 0.3, W * 0.4], head=('Где применено', 'Значение', 'Источник')))
    else:
        story.append(P('Не применялись: оценка по фото не выполнена, анализ крови не представлен.'))

    # 8. Ограничения
    story += [P('8. Ограничения метода', 'h2'), P(LIMITS_TEXT, 'small')]

    # 9. Отметки врача
    sig = Table([[P('Заключение врача:'), P('')], [P(''), P('')], [P(''), P('')],
                 [P('Врач (Ф. И. О., должность):'), P('')], [P('Подпись:'), P('Дата:')]],
                colWidths=[W * 0.5, W * 0.5], rowHeights=[16, 16, 16, 18, 22])
    sig.setStyle(TableStyle([('LINEBELOW', (0, 0), (-1, 3), 0.5, grid), ('LINEBELOW', (0, 4), (-1, 4), 0.5, grid),
                             ('SPAN', (0, 0), (0, 0)), ('VALIGN', (0, 0), (-1, -1), 'BOTTOM'),
                             ('LEFTPADDING', (0, 0), (-1, -1), 0)]))
    story += [KeepTogether([P('9. Отметки врача', 'h2'), sig])]

    # Приложение: Personalizer
    story.append(P('Приложение. Ответы AI Personalizer этой проверки', 'h2'))
    ent = personalizer_entries(reply, history)
    if ent:
        for who, text in ent:
            story.append(Paragraph(f'<b>{escape(who)}:</b> ' + escape(text).replace('\n', '<br/>'), st['p']))
            story.append(Spacer(1, 2))
        story.append(P('Personalizer только объясняет результат; уровень риска и Hb по фото он не меняет.', 'small'))
    else:
        story.append(P('Ответов Personalizer нет.'))

    class NumberedCanvas(rl_canvas.Canvas):
        """«стр. N из M»: страницы откладываются и подписываются, когда известно их число."""

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self._pages = []

        def showPage(self):
            self._pages.append(dict(self.__dict__))
            self._startPage()

        def save(self):
            n = len(self._pages)
            for page in self._pages:
                self.__dict__.update(page)
                self.saveState()
                self.setFont('HV', 7.5)
                self.setFillColor(muted)
                self.setStrokeColor(grid)
                self.line(18 * mm, 14 * mm, A4[0] - 18 * mm, 14 * mm)
                self.drawString(18 * mm, 10 * mm, DISCLAIMER)
                self.drawString(18 * mm, 6.5 * mm, f'HemaVision PoC · протокол № {rid} · {version}'[:110])
                self.drawRightString(A4[0] - 18 * mm, 10 * mm, f'стр. {self._pageNumber} из {n}')
                self.restoreState()
                super().showPage()
            super().save()

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm,
                            bottomMargin=18 * mm, title=TITLE, author='HemaVision PoC', subject=DISCLAIMER)
    doc.build(story, canvasmaker=NumberedCanvas)
    return path
