"""Отображение результата ядра для интерфейса HemaVision. Чистые функции, без gradio и без новых правил.

Анализ — anemia.predict.analyze_photo (как ячейка Я2); сводка качества — anemia.api.to_result; тексты уровней —
predict.RISK_TEXT; контуры масок — predict.overlay / overlay_photo. Здесь только перевод в HTML и тексты.
Медицинские правила отображения:
- уровень риска всегда словом и пояснением, цвет — дополнительно;
- нет оценки (status ≠ 'ok') → ни уровня, ни P, ни Hb, только причина и что сделать;
- Hb — только при status 'ok' и только если модель Hb прошла проверку (hb_range не None), с пометкой
  «экспериментально»;
- советы — только существующие тексты ядра (RISK_TEXT, сообщения проверок) и «сдайте общий анализ крови».
"""
import datetime as dt
import html
from dataclasses import dataclass, field

from anemia import api as API
from anemia import predict as PR

SEX = {'Женский': 'F', 'Мужской': 'M', 'Не указан': None}
PREG = {'Не указано': None, 'Нет': False, 'Да': True}
METHOD = {'Авто': None, 'Конъюнктива': 'conj', 'Склера': 'sclera', 'Обе ткани': 'both'}
ANSWER = {'Да': True, 'Нет': False, 'Не знаю': None}
BLOOD_TEST = 'Точно уровень гемоглобина показывает только общий анализ крови.'

LEVEL = {   # уровень → (css-класс, значок, подпись) — значок и подпись дублируют цвет
    'низкий': ('low', 'check', 'НИЗКИЙ РИСК'),
    'средний': ('mid', 'alert', 'СРЕДНИЙ РИСК'),
    'высокий': ('high', 'alert', 'ВЫСОКИЙ РИСК'),
    None: ('none', 'help', 'РИСК НЕ ОПРЕДЕЛЁН'),
}
STEPS = (   # этапы analyze_photo по порядку (подписи простым языком)
    ('age', 'Проверка возраста'),
    ('frame', 'Проверка фото: размер и свет'),
    ('eye', 'Поиск глаза'),
    ('tissue', 'Поиск слизистой века и белка глаза'),
    ('quality', 'Проверка качества снимка'),
    ('model', 'Сравнение с обучающими снимками и оценка риска'),
)
STOP_STEP = {'age': 0, 'frame': 1, 'no_eye': 2, 'no_tissue': 3, 'quality': 4, 'mismatch': 5, 'no_model': 5,
             'ood': 5}
STEP_STATE = {'done': ('check', 'выполнено'), 'fail': ('x', 'остановлено'), 'skip': ('minus', 'не выполнялось')}
TISSUE_PLAIN = {'обе': 'по цвету слизистой нижнего века и белка глаза', 'конъюнктива': 'по цвету слизистой нижнего века',
                'склера': 'по цвету белка глаза (склеры)'}
CV_NOTE = ('Показатели ниже получены ретроспективной кросс-валидацией на одном наборе данных (Eyes-defy, взрослые). '
           'Это не клиническая валидация и не доказательство точности на других телефонах и в других условиях.')
Q_TEXT = {'ok': 'в норме', 'размыто': 'размыто', 'темно': 'темно', 'пересвет': 'пересвет', None: 'не проверялось'}


@dataclass(frozen=True)
class Survey:
    sex: str | None = None
    age: float | None = None
    pregnant: bool | None = None
    method: str | None = None             # None = «Авто» (как в Я2), иначе predict.METHODS
    wellbeing: dict = field(default_factory=dict)


def parse_survey(sex, age, pregnant, method, answers=()):
    """Значения полей анкеты (подписи как в интерфейсе) → Survey. Возраст не проверяется здесь: это делает ядро
    (младше 5 лет — отказ, нераспознанный — отказ с причиной)."""
    from .personalizer import WELLBEING
    s = SEX.get(sex)
    age = None if age is None or age == '' else float(age)
    return Survey(sex=s, age=age, pregnant=PREG.get(pregnant) if s != 'M' else None, method=METHOD.get(method),
                  wellbeing={k: ANSWER.get(a) for (k, _), a in zip(WELLBEING, answers)})


# ------------------------------------------------------------------ модель и анализ
@dataclass
class Backend:
    bundle: dict | None = None
    landmarker: object = None
    version: str = ''
    error: str | None = None
    notes: tuple = ()


def load_backend(model_path=None, landmarker='auto'):
    """Модель ядра (Я1) + MediaPipe один раз при старте. Нет модели → Backend с error (интерфейс открывается и
    объясняет, что запустить), а не исключение."""
    try:
        bundle = PR.load_bundle(model_path)
    except (FileNotFoundError, RuntimeError) as e:
        return Backend(error=f'{e}')
    notes = ()
    if landmarker == 'auto':
        landmarker, note = PR.face_landmarker()
        notes = (note,)
    return Backend(bundle=bundle, landmarker=landmarker, version=API.model_version(bundle), notes=notes)


def run_analysis(photo, survey, backend):
    """Фото (путь / bytes / массив) + Survey → (r — dict analyze_photo, res — api.Result)."""
    r = PR.analyze_photo(photo, sex=survey.sex, age=survey.age, pregnant=survey.pregnant, bundle=backend.bundle,
                         landmarker=backend.landmarker, method=survey.method)
    return r, API.to_result(r, backend.version)


def overlays(r):
    """Картинки для экрана анализа: (вырезка глаза с контурами | None, всё фото лица с рамками | None)."""
    return PR.overlay(r), PR.overlay_photo(r)


# ------------------------------------------------------------------ HTML
ICONS = {   # контурные иконки 24×24 (stroke = currentColor), без внешних библиотек
    'check': '<path d="M20 6 9 17l-5-5"/>',
    'x': '<path d="M18 6 6 18M6 6l12 12"/>',
    'minus': '<path d="M5 12h14"/>',
    'alert': '<path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4M12 17h.01"/>',
    'help': '<circle cx="12" cy="12" r="10"/><path d="M9.1 9a3 3 0 0 1 5.8 1c0 2-3 3-3 3M12 17h.01"/>',
    'info': '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4M12 8h.01"/>',
    'sun': '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>',
    'eye': '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    'camera': '<path d="M14.5 4h-5L7.5 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9a2 2 0 0 0-2-2h-3.5z"/><circle cx="12" cy="13" r="3.5"/>',
    'focus': '<path d="M3 7V5a2 2 0 0 1 2-2h2M17 3h2a2 2 0 0 1 2 2v2M21 17v2a2 2 0 0 1-2 2h-2M7 21H5a2 2 0 0 1-2-2v-2"/><circle cx="12" cy="12" r="3"/>',
    'file': '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h5"/>',
    'clipboard': '<rect x="8" y="2" width="8" height="4" rx="1"/><path d="M16 4h2a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h2M9 12h6M9 16h4"/>',
    'shield': '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    'drop': '<path d="M12 2.7s6 6.3 6 11.3a6 6 0 0 1-12 0c0-5 6-11.3 6-11.3z"/>',
}


def icon(name):
    """Иконка как встроенный SVG (декоративная: aria-hidden, смысл всегда дублируется текстом)."""
    return ('<span class="hv-ic" aria-hidden="true"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round">{ICONS[name]}</svg></span>')


def _e(x):
    return html.escape(str(x))


def _cap(s):
    s = str(s)
    return s[:1].upper() + s[1:]


def note_html(kind, body):
    """Плашка: kind 'info' | 'warn' | 'ok' | 'err' (иконка + текст; цвет только дополняет)."""
    cls, ic = {'info': ('hv-info', 'info'), 'warn': ('hv-warn', 'alert'), 'ok': ('hv-okline', 'check'),
               'err': ('hv-err', 'x')}[kind]
    return f'<div class="hv-note {cls}" role="{"alert" if kind == "err" else "note"}">{icon(ic)}<div>{body}</div></div>'


def model_status_html(backend):
    """Главная: готово ли приложение. Версия модели — в подвале и в разделе для врача, не здесь."""
    if backend.error:
        return note_html('warn', '<b>Анализ пока недоступен: модель не загружена.</b> Запустите ячейку Я1 (обучение '
                                 f'модели) и перезапустите интерфейс.<br><span class="hv-small">{_e(backend.error)}</span>')
    return note_html('ok', '<b>Готово к работе.</b> Модель загружена, можно начинать.')


def footer_html(backend):
    v = backend.version or 'модель не загружена'
    return f'<p class="hv-footer">HemaVision PoC · {_e(v)} · скрининг риска, не диагноз</p>'


def stepper_html(active, labels=('Фото', 'Анкета', 'Анализ', 'Результат')):
    """Индикатор шага мастера: active — номер 1..4 (0 — главная, индикатор скрыт)."""
    if not active:
        return ''
    items = []
    for i, name in enumerate(labels, 1):
        cls = 'done' if i < active else 'cur' if i == active else ''
        cur = ' aria-current="step"' if i == active else ''
        items.append(f'<li class="{cls}"{cur}><span class="n">{i}</span><span class="t">{_e(name)}</span></li>')
    return (f'<nav aria-label="Шаг {active} из {len(labels)}"><ol class="hv-stepper">{"".join(items)}</ol></nav>')


def title_html(title, sub=''):
    return f'<h2 class="hv-h2">{_e(title)}</h2>' + (f'<p class="hv-sub">{_e(sub)}</p>' if sub else '')


def photo_tips_html():
    """Инструкция перед съёмкой (видна сразу, не в свёрнутом блоке)."""
    tips = (('sun', 'Дневной свет у окна', 'Без прямого солнца, без вспышки и без фильтров камеры.'),
            ('eye', 'Оттяните нижнее веко', 'Пальцем, взгляд вверх: розовая слизистая видна полосой.'),
            ('focus', 'Близко и резко', 'Телефон в 10–15 см от глаза, коснитесь экрана на веке для фокуса.'),
            ('camera', 'Оригинал из камеры', 'Не скриншот и не фото, пересланное через мессенджер.'))
    li = ''.join(f'<li><span class="ti">{icon(i)}</span><div><b>{_e(t)}</b><br><span class="hv-small">{_e(d)}</span>'
                 f'</div></li>' for i, t, d in tips)
    return f'<div class="hv-card"><h3>Как сделать фото</h3><ul class="hv-tips">{li}</ul></div>'


def loading_html():
    return ('<div class="hv-card" role="status" aria-live="polite"><div class="hv-loading"><span class="hv-spin" '
            'aria-hidden="true"></span><div><b>Идёт анализ снимка…</b><br><span class="hv-small">Ищем глаз, '
            'проверяем качество фото и оцениваем цвет. Не закрывайте страницу.</span></div></div></div>')


def step_states(r):
    """Этапы конвейера → [(подпись, 'done' | 'fail' | 'skip')] по stage, на котором остановилось ядро."""
    stop = STOP_STEP.get(r.get('stage'))
    if r.get('status') == 'ok' or stop is None:
        return [(label, 'done') for _, label in STEPS]
    return [(label, 'done' if i < stop else 'fail' if i == stop else 'skip') for i, (_, label) in enumerate(STEPS)]


def pipeline_html(r, seconds=None):
    rows = []
    for label, st in step_states(r):
        ic, word = STEP_STATE[st]
        rows.append(f'<li class="hv-step {st}"><span class="ic">{icon(ic)}</span>'
                    f'<span class="lb">{_e(label)}</span><span class="st">{word}</span></li>')
    head = ('Анализ завершён: снимок прошёл все проверки.' if r.get('status') == 'ok' else
            f'Анализ остановлен: {_e(PR.STATUS_RU.get(r.get("status"), r.get("status")))}.')
    t = f'<p class="hv-small">Время анализа: {seconds:.1f} с</p>' if seconds is not None else ''
    return (f'<div class="hv-card" role="status" aria-live="polite"><p class="hv-lead">{head}</p>'
            f'<ul class="hv-pipe">{"".join(rows)}</ul>{t}</div>')


def risk_scale_html(risk):
    segs = []
    for lv in ('низкий', 'средний', 'высокий'):
        cls, _, _ = LEVEL[lv]
        on = risk == lv
        segs.append(f'<div class="seg {cls}{" on" if on else ""}"><span>{lv.capitalize()}</span>'
                    f'{"<b>▲ ваш результат</b>" if on else ""}</div>')
    return f'<div class="hv-scale" role="img" aria-label="Шкала риска: {_e(risk or "не определён")}">{"".join(segs)}</div>'


def risk_card_html(r):
    """Главная карточка результата. Нет оценки → причина и что сделать, без уровня и без цифр."""
    ok = r.get('status') == 'ok'
    risk = r.get('risk') if ok else None
    cls, ic, title = LEVEL.get(risk, LEVEL[None])
    if ok:
        body = f'<p class="hv-explain">{_e(_cap(PR.RISK_TEXT[risk]))}.</p>'
    else:
        issues = ''.join(f'<li>{_e(x)}</li>' for x in r.get('issues') or []) or '<li>причина не указана</li>'
        body = (f'<p class="hv-explain"><b>{_e(_cap(PR.STATUS_RU.get(r.get("status"), "оценка не выполнена")))}.'
                f'</b> Уровень риска и гемоглобин не выдаются.</p><ul class="hv-issues">{issues}</ul>'
                f'<p class="hv-explain">{"" if r.get("stage") == "age" else "Переснимите фото по подсказкам на шаге «Фото». "}'
                f'{BLOOD_TEST}</p>')
    return (f'<div class="hv-card hv-risk {cls}" role="status" aria-live="polite"><div class="hv-badge">{icon(ic)}'
            f'<span>{title}</span></div>{body}{risk_scale_html(risk)}</div>')


def hb_html(r):
    """Hb для пользователя: только при оценке и только если модель Hb прошла проверку. Технические цифры проверки
    модели Hb — в разделе для врача (doctor_html). Без оценки — пустая строка."""
    if r.get('status') != 'ok':
        return ''
    if r.get('hb_range'):
        lo, hi = r['hb_range']
        return (f'<div class="hv-card"><h3>{icon("drop")} Ориентировочный гемоглобин <span class="hv-tag">экспериментально'
                f'</span></h3><p class="hv-hb">{lo:.0f}–{hi:.0f} <small>г/л</small></p><p class="hv-small">Это грубый '
                f'ориентир по цвету на фото, а не результат анализа. {BLOOD_TEST}</p></div>')
    return (f'<div class="hv-card"><h3>{icon("drop")} Гемоглобин</h3><p class="hv-small">По этому фото цифра гемоглобина '
            f'не показывается: модель Hb пока недостаточно точна. {BLOOD_TEST}</p></div>')


def quality_html(res):
    q = res.quality
    vis = q.get('tissue_visible') or {}

    def tv(v):
        return ('na', 'не проверялось') if v is None else ('ok', 'найдена') if v else ('bad', 'не найдена')

    def qv(v):
        return ('na' if v is None else 'ok' if v == 'ok' else 'bad', Q_TEXT.get(v, v))
    rows = [('Резкость', qv(q.get('sharpness'))), ('Свет', qv(q.get('light'))),
            ('Слизистая века (конъюнктива)', tv(vis.get('conjunctiva'))), ('Белок глаза (склера)', tv(vis.get('sclera')))]
    if res.tissue:
        rows.append(('Оценено по', ('ok', res.tissue)))
    ic = {'ok': 'check', 'bad': 'x', 'na': 'minus'}
    li = ''.join(f'<li><span>{_e(a)}</span><span class="v {k}">{icon(ic[k])}{_e(b)}</span></li>' for a, (k, b) in rows)
    return f'<div class="hv-card"><h3>Качество снимка</h3><ul class="hv-qlist">{li}</ul></div>'


def summary_html(r, res=None):
    """«Коротко» простым языком: по какой ткани оценено и примечания ядра. Без P, порогов и метрик."""
    if r.get('status') == 'ok':
        t = TISSUE_PLAIN.get(res.tissue if res is not None else None, 'по цвету на фото')
        lines = [f'Оценено {t}. Оценка показывает риск, а не диагноз.']
    else:
        lines = ['Оценка не выполнена: причины перечислены в карточке выше.']
    lines += [_cap(x) for x in r.get('notes') or []]
    return '<div class="hv-card"><h3>Коротко</h3>' + ''.join(f'<p>{_e(x)}</p>' for x in lines) + '</div>'


def doctor_html(r, version=''):
    """Раздел «Для врача»: технические сведения ядра. С оговоркой: это не клиническая валидация."""
    ok = r.get('status') == 'ok'
    items = [f'Статус: {_e(PR.STATUS_RU.get(r.get("status"), r.get("status")))} (этап {_e(r.get("stage"))})',
             f'Модель: {_e(r.get("model") or "—")}', f'Версия: {_e(version or "—")}']
    if r.get('method'):
        items.append(f'Выбор ткани: {_e(PR.METHOD_RU.get(r["method"], "Авто"))}')
    items += [_e(x) for x in (r.get('reasons') if ok else r.get('issues')) or []]
    if ok and r.get('hb_note'):
        items.append('Hb: ' + _e(r['hb_note']))
    items += ['Примечание: ' + _e(x) for x in r.get('notes') or []]
    diag = ''.join(f'<li>{_e(x)}</li>' for x in r.get('diag') or [])
    return (f'<div class="hv-doc">{note_html("info", _e(CV_NOTE))}<ul>{"".join(f"<li>{x}</li>" for x in items)}</ul>'
            + (f'<p><b>Диагностика ядра</b></p><ul>{diag}</ul>' if diag else '') + '</div>')


def legend_html(crop=True, face=False):
    """Подпись к картинкам экрана анализа (цвета контуров словами)."""
    parts = []
    if crop:
        parts.append('<b>Найденные ткани:</b> зелёный контур — слизистая нижнего века, голубой — белок глаза.')
    if face:
        parts.append('<b>Глаза на фото:</b> зелёная рамка — глаз принят, красная — не подошёл.')
    return '<p class="hv-small">' + '<br>'.join(parts) + '</p>' if parts else ''


def error_html(e):
    """Непредвиденная ошибка кода: без какого-либо результата."""
    return risk_card_html({'status': 'unreliable', 'stage': 'frame',
                           'issues': [f'ошибка программы при анализе: {type(e).__name__}: {e}']})


def reasons_md(r):
    """«Почему такой ответ» (Markdown): объяснения ядра, примечания, дисклеймер."""
    lines = []
    if r.get('status') == 'ok':
        lines += [f'- {x}' for x in r.get('reasons') or []]
    else:
        lines += [f'- {x}' for x in r.get('issues') or []]
    lines += [f'- *Примечание:* {x}' for x in r.get('notes') or []]
    if r.get('model'):
        lines.append(f'- Модель: {r["model"]}')
    return '\n'.join(lines) if lines else '—'


def report_text(r, survey, backend_version=''):
    """Отчёт для врача (.txt): результат ядра (predict.format_result) + анкета со слов пользователя."""
    ru = {True: 'да', False: 'нет', None: 'не указано'}
    from .personalizer import WELLBEING
    head = ['HemaVision — отчёт скрининга риска анемии по фото глаза (не диагноз)',
            f'Дата: {dt.datetime.now().strftime("%Y-%m-%d %H:%M")}', f'Версия модели: {backend_version or "—"}', '',
            'Анкета (со слов пользователя):',
            (f'  пол: {({"F": "женский", "M": "мужской"}).get(survey.sex, "не указан")}; '
             f'возраст: {"не указан" if survey.age is None else f"{survey.age:g}"}; беременность: {ru[survey.pregnant]}'),
            f'  ткань: {PR.METHOD_RU.get(survey.method, "Авто")}']
    head += [f'  {q} — {ru[survey.wellbeing.get(k)]}' for k, q in WELLBEING]
    head += ['  (ответы о самочувствии на оценку риска по фото не влияют)', '', CV_NOTE, '', 'Результат:']
    return '\n'.join(head) + '\n' + PR.format_result(r) + '\n'
