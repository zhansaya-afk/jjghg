"""PDF-отчёт HemaVision для пациента и врача (A4, офлайн, reportlab). Только отображение: без новых правил.

Источник данных — результат текущей проверки: r (predict.analyze_photo), res (api.to_result), анкета (view.Survey),
готовый анализ крови (labs.lab_hb_report, как в интерфейсе), ответы AI Personalizer этой сессии. Ничего не
пересчитывается: уровень риска, Hb, тексты причин и пояснений — из ядра; нормы — те же функции labs, что использует
ядро. Отсутствующие данные печатаются как «нет данных» / «не указано» / «не оценивался», демонстрационные значения
не подставляются никогда.
Правила как на экране результата: status ≠ 'ok' → ни уровня, ни Hb; Hb — только если ядро выдало hb_range, с
неопределённостью из ядра и пометкой «не лабораторное измерение»; анализ крови — отдельным разделом.

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
TITLE = 'HemaVision — отчёт скрининга риска анемии по фото глаза'
NO_DATA = 'нет данных'
WHO2011_SOURCE = ('ВОЗ 2011: WHO, Haemoglobin concentrations for the diagnosis of anaemia and assessment of '
                  'severity (WHO/NMH/NHD/MNM/11.1), таблица 1')
LEVELS = ('низкий', 'средний', 'высокий')
LEVEL_COLORS = {'низкий': ('#166534', '#f0fdf4'), 'средний': ('#92400e', '#fffbeb'), 'высокий': ('#b91c1c', '#fef2f2'),
                None: ('#334155', '#f1f5f9')}
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
        lo, hi = r['hb_range']
        note = r.get('hb_note') or 'неопределённость модели не указана'
        return (f'Ориентировочный Hb по фото: {lo:.0f}–{hi:.0f} г/л',
                f'Это НЕ лабораторное измерение, а грубый ориентир по цвету на фото. Неопределённость: {note}.')
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


# ------------------------------------------------------------------ сборка PDF
def build_pdf(path, r, survey, *, res=None, lab=None, photo=None, model_version='', reply=None, history=(),
              created=None):
    """Записать PDF-отчёт в path → path. r — dict analyze_photo (полный, частичный или отказ), survey — view.Survey,
    res — api.Result (качество) или None, lab — labs.lab_hb_report или None, photo — исходное фото или None,
    reply / history — ответы Personalizer этой проверки (или None / ())."""
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import mm
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
    ink, muted, line, accent = (colors.HexColor(c) for c in ('#0f172a', '#475569', '#cbd5e1', '#0e7490'))
    st = {
        'h1': ParagraphStyle('h1', fontName='HV-Bold', fontSize=15, leading=19, textColor=accent),
        'h2': ParagraphStyle('h2', fontName='HV-Bold', fontSize=11.5, leading=15, textColor=ink, spaceBefore=8,
                             spaceAfter=3),
        'p': ParagraphStyle('p', fontName='HV', fontSize=10, leading=13.5, textColor=ink),
        'small': ParagraphStyle('small', fontName='HV', fontSize=9, leading=12, textColor=muted),
        'cap': ParagraphStyle('cap', fontName='HV', fontSize=9, leading=11, textColor=muted, alignment=TA_CENTER),
        'risk': ParagraphStyle('risk', fontName='HV-Bold', fontSize=17, leading=21),
    }

    def P(text, style='p'):
        return Paragraph(escape(str(text)).replace('\n', '<br/>'), st[style])

    def B(label, value, style='p'):
        return Paragraph(f'<b>{escape(str(label))}:</b> {escape(str(value))}', st[style])

    def table(rows, widths, head=None):
        data = ([[Paragraph(f'<b>{escape(h)}</b>', st['small']) for h in head]] if head else [])
        data += [[P(c) for c in row] for row in rows]
        t = Table(data, colWidths=widths)
        t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.6, line), ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                               ('LEFTPADDING', (0, 0), (-1, -1), 4), ('RIGHTPADDING', (0, 0), (-1, -1), 4),
                               ('TOPPADDING', (0, 0), (-1, -1), 2), ('BOTTOMPADDING', (0, 0), (-1, -1), 3)]
                              + ([('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9'))] if head else [])))
        return t

    W = A4[0] - 36 * mm
    story = [P(TITLE, 'h1'),
             P(f'Дата и время: {created:%d.%m.%Y %H:%M} · отчёт № {rid} · версия модели: {version}', 'small'),
             Spacer(1, 4)]

    disc = Table([[Paragraph(f'<b>{escape(DISCLAIMER)}</b> Оценка риска по цвету глаза на фото; есть ли анемия, '
                             'показывает только общий анализ крови.', st['p'])]], colWidths=[W])
    disc.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#334155')),
                              ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
                              ('LEFTPADDING', (0, 0), (-1, -1), 8), ('TOPPADDING', (0, 0), (-1, -1), 6),
                              ('BOTTOMPADDING', (0, 0), (-1, -1), 7)]))
    story += [disc, Spacer(1, 4),
              P('Пациент (заполняет сам): ________________________________   Дата рождения: ______________', 'small')]

    # 1. Результат
    rb = risk_block(r)
    fg, bg = LEVEL_COLORS[rb['level']]
    st['risk'].textColor = colors.HexColor(fg)
    story.append(P('1. Результат оценки риска по фото', 'h2'))
    cells = []
    for lv in LEVELS:
        on = lv == rb['level']
        cells.append(Paragraph(('<b>' if on else '') + escape(_cap(lv)) + (' ▲ ваш результат</b>' if on else ''),
                               st['p']))
    scale = Table([cells], colWidths=[(W - 16) / 3] * 3)
    scale_style = [('BOX', (i, 0), (i, 0), 0.6, line) for i in range(3)]
    if rb['level']:
        i = LEVELS.index(rb['level'])
        scale_style += [('BOX', (i, 0), (i, 0), 2, colors.HexColor(fg)),
                        ('BACKGROUND', (i, 0), (i, 0), colors.HexColor(bg))]
    scale.setStyle(TableStyle(scale_style))
    box = [Paragraph(escape(rb['title']), st['risk']), Spacer(1, 3), scale, Spacer(1, 4), P(rb['text'])]
    box += [P('• ' + x) for x in rb['issues']]
    box += [Spacer(1, 3), P('Что сделать: сдайте общий анализ крови (гемоглобин) и покажите этот отчёт врачу. '
                            'Точно уровень гемоглобина показывает только анализ крови.')]
    frame = Table([[box]], colWidths=[W])
    frame.setStyle(TableStyle([('BOX', (0, 0), (-1, -1), 1, colors.HexColor(fg)),
                               ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor(bg)),
                               ('LEFTPADDING', (0, 0), (-1, -1), 8), ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                               ('TOPPADDING', (0, 0), (-1, -1), 6), ('BOTTOMPADDING', (0, 0), (-1, -1), 8)]))
    story.append(frame)

    story += [P('• Примечание: ' + str(x)) for x in r.get('notes') or []]

    # 2. Hb
    t, txt = hb_block(r)
    story += [P('2. Гемоглобин по фото', 'h2'), Paragraph(f'<b>{escape(t)}</b>', st['p']), P(txt)]

    # 3. Фото и маска
    story.append(P('3. Фото и найденные ткани', 'h2'))
    ims = images(r, photo)
    if ims:
        cw = W / len(ims)
        row_im, row_cap = [], []
        for cap, im in ims:
            w = min(cw - 6, 70 * mm)
            h = w * im.height / im.width
            if h > 60 * mm:
                h, w = 60 * mm, 60 * mm * im.width / im.height
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=85)
            buf.seek(0)
            row_im.append(Image(buf, width=w, height=h))
            row_cap.append(Paragraph(escape(cap), st['cap']))
        gal = Table([row_im, row_cap], colWidths=[cw] * len(ims))
        gal.setStyle(TableStyle([('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        story.append(KeepTogether([gal]))
        if any(c.startswith('Найденные') for c, _ in ims):
            story.append(P('Зелёный контур — слизистая нижнего века, голубой — белок глаза (склера): только эти '
                           'области вошли в расчёт.', 'small'))
        if any(c == 'Глаза на фото' for c, _ in ims):
            story.append(P('Зелёная рамка — глаз принят, красная — не подошёл.', 'small'))
    else:
        story.append(P('Изображения недоступны: ' + NO_DATA + '.', 'small'))

    # 4. Качество
    story += [P('4. Качество фотографии', 'h2'), table(quality_rows(res), [W * 0.55, W * 0.45])]

    # 5. Анкета и анализ крови
    story += [P('5. Анкета (со слов пользователя)', 'h2'), table(survey_rows(survey), [W * 0.62, W * 0.38]),
              P('Ответы о самочувствии на оценку риска по фото не влияют.', 'small'),
              P('Анализ крови (введён пользователем, отдельно от оценки по фото)', 'h2'),
              P(lab['text'] if lab else 'Не введён.')]

    # 6. Референсные значения
    story.append(P('6. Применённые референсные значения', 'h2'))
    refs = references(r, survey, lab)
    if refs:
        story.append(table(refs, [W * 0.3, W * 0.3, W * 0.4], head=('Где применено', 'Значение', 'Источник')))
    else:
        story.append(P('Не применялись: оценка по фото не выполнена, анализ крови не введён.'))

    # 7. Personalizer
    story.append(P('7. AI Personalizer (объяснение и вопросы этой проверки)', 'h2'))
    ent = personalizer_entries(reply, history)
    if ent:
        for who, text in ent:
            story.append(Paragraph(f'<b>{escape(who)}:</b> ' + escape(text).replace('\n', '<br/>'), st['p']))
            story.append(Spacer(1, 2))
        story.append(P('Personalizer только объясняет результат; риск и Hb по фото он не меняет.', 'small'))
    else:
        story.append(P('Ответов Personalizer нет.'))

    # 8. Для врача
    story += [P('8. Для врача: технические сведения', 'h2'),
              B('Статус', f'{PR.STATUS_RU.get(r.get("status"), r.get("status") or NO_DATA)} '
                          f'(этап: {r.get("stage") or NO_DATA})', 'small'),
              B('Модель', r.get('model') or NO_DATA, 'small'), B('Версия модели', version, 'small')]
    why = r.get('reasons') if r.get('status') == 'ok' else []
    story += [P('Почему такой ответ (объяснение ядра): ' + '; '.join(str(x) for x in why), 'small')] if why else []
    story += [P('Показатели модели получены ретроспективной кросс-валидацией на одном наборе данных (Eyes-defy: '
                'взрослые Италии и Индии, снимки через насадку). Это не клиническая валидация; точность на других '
                'телефонах, в других условиях и у жителей Казахстана не проверена.', 'small'),
              Spacer(1, 6), Paragraph(f'<b>{escape(DISCLAIMER)}</b>', st['p'])]

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('HV', 8)
        canvas.setFillColor(muted)
        canvas.drawString(18 * mm, 10 * mm, DISCLAIMER)
        canvas.drawString(18 * mm, 6.5 * mm, f'HemaVision PoC · {rid} · {version}'[:110])
        canvas.drawRightString(A4[0] - 18 * mm, 10 * mm, f'стр. {doc.page}')
        canvas.restoreState()

    doc = SimpleDocTemplate(path, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=14 * mm,
                            bottomMargin=16 * mm, title=TITLE, author='HemaVision PoC', subject=DISCLAIMER)
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    return path
