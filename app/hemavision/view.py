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
    'низкий': ('low', '✓', 'НИЗКИЙ РИСК'),
    'средний': ('mid', '!', 'СРЕДНИЙ РИСК'),
    'высокий': ('high', '!!', 'ВЫСОКИЙ РИСК'),
    None: ('none', '?', 'РИСК НЕ ОПРЕДЕЛЁН'),
}
STEPS = (   # этапы analyze_photo по порядку
    ('age', 'Анкета: возраст подходит для оценки'),
    ('frame', 'Кадр: файл, размер, свет'),
    ('eye', 'Поиск глаза'),
    ('tissue', 'Поиск конъюнктивы и склеры'),
    ('quality', 'Качество тканей'),
    ('model', 'Модель риска и проверка «снимок похож на обучающие»'),
)
STOP_STEP = {'age': 0, 'frame': 1, 'no_eye': 2, 'no_tissue': 3, 'quality': 4, 'mismatch': 5, 'no_model': 5,
             'ood': 5}
STEP_STATE = {'done': ('✓', 'выполнено'), 'fail': ('✕', 'остановлено'), 'skip': ('–', 'не выполнялось')}
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
def _e(x):
    return html.escape(str(x))


def _cap(s):
    s = str(s)
    return s[:1].upper() + s[1:]


def model_status_html(backend):
    if backend.error:
        return (f'<div class="hv-note hv-warn"><b>Модель не загружена.</b> Анализ недоступен: запустите ячейку Я1 '
                f'(обучение модели ядра) и перезапустите интерфейс.<br><small>{_e(backend.error)}</small></div>')
    extra = ''.join(f'<br><small>{_e(n)}</small>' for n in backend.notes if n)
    return f'<div class="hv-note hv-okline"><b>Модель загружена:</b> {_e(backend.version)}{extra}</div>'


def stepper_html(active, labels=('Фото', 'Анкета', 'Анализ', 'Результат')):
    """Индикатор шага мастера: active — номер 1..4 (0 — главная, индикатор скрыт)."""
    if not active:
        return ''
    items = []
    for i, name in enumerate(labels, 1):
        cls = 'done' if i < active else 'cur' if i == active else ''
        items.append(f'<li class="{cls}"><span>{i}</span>{_e(name)}</li>')
    return f'<ol class="hv-stepper" aria-label="Шаг {active} из {len(labels)}">{"".join(items)}</ol>'


def step_states(r):
    """Этапы конвейера → [(подпись, 'done' | 'fail' | 'skip')] по stage, на котором остановилось ядро."""
    stop = STOP_STEP.get(r.get('stage'))
    if r.get('status') == 'ok' or stop is None:
        return [(label, 'done') for _, label in STEPS]
    return [(label, 'done' if i < stop else 'fail' if i == stop else 'skip') for i, (_, label) in enumerate(STEPS)]


def pipeline_html(r):
    rows = []
    for label, st in step_states(r):
        icon, word = STEP_STATE[st]
        rows.append(f'<li class="hv-step {st}"><span class="ic" aria-hidden="true">{icon}</span>'
                    f'<span class="lb">{_e(label)}</span><span class="st">{word}</span></li>')
    head = ('Анализ завершён: снимок прошёл все проверки.' if r.get('status') == 'ok' else
            f'Анализ остановлен: {_e(PR.STATUS_RU.get(r.get("status"), r.get("status")))}.')
    return f'<div class="hv-card"><p class="hv-lead">{head}</p><ul class="hv-steps">{"".join(rows)}</ul></div>'


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
    cls, icon, title = LEVEL.get(risk, LEVEL[None])
    if ok:
        body = f'<p class="hv-explain">{_e(_cap(PR.RISK_TEXT[risk]))}.</p>'
    else:
        issues = ''.join(f'<li>{_e(x)}</li>' for x in r.get('issues') or []) or '<li>причина не указана</li>'
        body = (f'<p class="hv-explain"><b>{_e(_cap(PR.STATUS_RU.get(r.get("status"), "оценка не выполнена")))}.'
                f'</b> Уровень риска и гемоглобин не выдаются.</p><ul class="hv-issues">{issues}</ul>'
                f'<p class="hv-explain">{"" if r.get("stage") == "age" else "Переснимите по подсказкам выше. "}'
                f'{BLOOD_TEST}</p>')
    return (f'<div class="hv-card hv-risk {cls}"><div class="hv-badge"><span class="ic" aria-hidden="true">{icon}'
            f'</span><span>{title}</span></div>{body}{risk_scale_html(risk)}</div>')


def hb_html(r):
    """Hb: только при оценке и только если модель Hb прошла проверку. Без оценки — пустая строка."""
    if r.get('status') != 'ok':
        return ''
    if r.get('hb_range'):
        lo, hi = r['hb_range']
        return (f'<div class="hv-card"><h3>Ориентировочный гемоглобин <span class="hv-tag">экспериментально</span></h3>'
                f'<p class="hv-hb">{lo:.0f}–{hi:.0f} <small>г/л</small></p><p class="hv-small">{_e(r.get("hb_note") or "")}. '
                f'Это не результат анализа. {BLOOD_TEST}</p></div>')
    return (f'<div class="hv-card"><h3>Гемоглобин</h3><p class="hv-small">{_e(r.get("hb_note") or "цифра Hb не выдаётся")}. '
            f'{BLOOD_TEST}</p></div>')


def quality_html(res):
    q = res.quality
    vis = q.get('tissue_visible') or {}

    def tv(v):
        return 'не проверялось' if v is None else 'найдена' if v else 'не найдена'
    rows = [('Резкость', Q_TEXT.get(q.get('sharpness'), q.get('sharpness'))),
            ('Свет', Q_TEXT.get(q.get('light'), q.get('light'))),
            ('Конъюнктива', tv(vis.get('conjunctiva'))), ('Склера', tv(vis.get('sclera')))]
    if res.tissue:
        rows.append(('Оценено по', res.tissue))
    trs = ''.join(f'<tr><td>{_e(a)}</td><td>{_e(b)}</td></tr>' for a, b in rows)
    return f'<div class="hv-card"><h3>Качество снимка</h3><table class="hv-table">{trs}</table></div>'


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
    head += ['  (ответы о самочувствии на оценку риска по фото не влияют)', '', 'Результат:']
    return '\n'.join(head) + '\n' + PR.format_result(r) + '\n'
