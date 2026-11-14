"""Раскладка интерфейса HemaVision (Gradio 6): мастер из 5 экранов.

Главная → Фото → Анкета → Анализ → Результат (+ блок AI Personalizer). Экраны — колонки, видна одна;
логика отображения — view.py, персонализатор — personalizer.py. Добавить экран: колонка в SCREENS + кнопки.
"""
import inspect
import os
import tempfile
import time

from . import view as V
from .personalizer import WELLBEING, Personalizer, RulesPersonalizer, make_input
from .theme import CSS, LIGHT_JS, theme

assert len(WELLBEING) == 3, 'run() принимает 3 ответа о самочувствии — поправьте сигнатуру'
SCREENS = ('home', 'photo', 'survey', 'analysis', 'result')
STEP_OF = {'home': 0, 'photo': 1, 'survey': 2, 'analysis': 3, 'result': 4}
DISCLAIMER_HTML = V.note_html('info', '<b>HemaVision — предварительный скрининг, не диагностика.</b> Приложение '
                              'оценивает риск анемии по цвету слизистой века и белка глаза. Диагноз ставит врач по '
                              'общему анализу крови.')
HOME_HTML = """
<h1 class="hv-h1">HemaVision</h1>
<p class="hv-sub">Проверка риска анемии по фото глаза: займёт несколько минут. Фото обрабатывается в этой среде,
без облачных сервисов анализа, и не сохраняется в данные проекта.</p>
"""
PHOTO_TIPS = """**Как снимать**
- Дневной свет у окна, без прямого солнца, **без вспышки** и фильтров камеры.
- Оттяните нижнее веко пальцем, смотрите вверх: розовая слизистая должна быть видна полосой.
- Телефон в 10–15 см от глаза, фокус тапом по веку.
- Загружайте **оригинал из камеры**: не скриншот и не фото из мессенджера."""
PHOTO_HELP = ('<p class="hv-small">Нажмите на поле, чтобы выбрать готовое фото. На телефоне система может предложить '
              'галерею или камеру — это зависит от телефона и браузера. Значок камеры под полем включает камеру '
              'браузера (удобно на ноутбуке; на телефоне снимок обычно хуже, чем в приложении «Камера»).</p>')
MIN_COL = 320               # px: уже — колонки ряда встают друг под другом (телефон, узкий планшет)


def _home_steps():
    steps = (('camera', 'Фото глаза', 'С оттянутым нижним веком, при дневном свете.'),
             ('clipboard', 'Короткая анкета', 'Пол и возраст — для нормы гемоглобина ВОЗ.'),
             ('file', 'Результат и отчёт', 'Шкала риска с пояснением и отчёт для врача.'))
    li = ''.join(f'<li><span class="num">{V.icon(i)}</span><div><b>{t}</b><span class="hv-small">{d}</span></div></li>'
                 for i, t, d in steps)
    return f'<div class="hv-card"><h3>Как это работает</h3><ol class="hv-steps3">{li}</ol></div>'


def _personalizer_html(reply):
    import re
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', V._e(reply.text)).replace('\n\n', '</p><p>')
    return f'<div class="hv-card hv-personalizer"><h3>{V._e(reply.title)}</h3><p class="hv-small">{text}</p></div>'


def _blocks_kwargs(gr):
    """Gradio 6 принимает theme/css/js в launch(), Gradio 5 — в Blocks(). Возвращает (для Blocks, для launch)."""
    style = {'theme': theme(), 'css': CSS, 'js': LIGHT_JS}
    if 'css' in inspect.signature(gr.Blocks.__init__).parameters:
        return style, {}
    if 'footer_links' in inspect.signature(gr.Blocks.launch).parameters:
        style['footer_links'] = []          # без «API · Gradio · Настройки» в подвале
    return {}, style


def build(model_path=None, personalizer: Personalizer | None = None, backend=None):
    """→ gr.Blocks. backend — готовый view.Backend (для тестов); иначе модель Я1 грузится здесь один раз.
    personalizer по умолчанию — RulesPersonalizer (офлайн). Телеметрия Gradio выключена: приложение офлайн."""
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'     # до импорта: Gradio не шлёт статистику и не проверяет версию
    import gradio as gr

    backend = backend if backend is not None else V.load_backend(model_path)
    personalizer = personalizer or RulesPersonalizer()
    blocks_kw, launch_kw = _blocks_kwargs(gr)

    def show(name):
        return [gr.update(visible=s == name) for s in SCREENS] + [V.stepper_html(STEP_OF[name])]

    with gr.Blocks(title='HemaVision — скрининг риска анемии', elem_classes='hv-app', analytics_enabled=False,
                   **blocks_kw) as demo:
        demo.hv_launch_kwargs = launch_kw
        state = gr.State(None)            # {'r': dict analyze_photo, 'survey': Survey}
        stepper = gr.HTML('')

        # ---------------------------------------------------------------- 0. Главная
        with gr.Column(visible=True) as home:
            with gr.Row(equal_height=False):
                with gr.Column(scale=3, min_width=MIN_COL):
                    gr.HTML(HOME_HTML + _home_steps())
                with gr.Column(scale=2, min_width=MIN_COL):
                    gr.HTML(DISCLAIMER_HTML + V.model_status_html(backend))
                    start = gr.Button('Начать проверку', variant='primary', size='lg')

        # ---------------------------------------------------------------- 1. Фото
        with gr.Column(visible=False) as photo_scr:
            gr.HTML(V.title_html('Фото глаза', 'Сначала прочитайте, как снимать: от качества фото зависит, будет ли '
                                               'оценка вообще.'))
            with gr.Row(equal_height=False):
                with gr.Column(min_width=MIN_COL):
                    gr.HTML(V.photo_tips_html())
                with gr.Column(min_width=MIN_COL):
                    photo = gr.Image(type='filepath', image_mode=None, sources=['upload', 'webcam'],
                                     label='Фото глаза', height=340, webcam_options=gr.WebcamOptions(mirror=False),
                                     placeholder='Нажмите, чтобы выбрать фото, или перетащите его сюда')
                    gr.HTML(PHOTO_HELP)
                    photo_msg = gr.HTML()
            with gr.Row(elem_classes='hv-nav'):
                back1 = gr.Button('← Назад', size='lg')
                next1 = gr.Button('Далее: анкета →', variant='primary', size='lg')

        # ---------------------------------------------------------------- 2. Анкета
        with gr.Column(visible=False) as survey_scr:
            gr.HTML(V.title_html('Анкета', 'Пол, возраст и беременность нужны для нормы гемоглобина ВОЗ. Детям младше '
                                           '5 лет оценка по фото не проводится.'))
            with gr.Row(equal_height=False):
                with gr.Column(min_width=MIN_COL):
                    sex = gr.Radio(list(V.SEX), value='Женский', label='Пол')
                    age = gr.Number(label='Возраст, полных лет', value=None, minimum=0, maximum=120, precision=0)
                    preg = gr.Radio(list(V.PREG), value='Не указано', label='Беременность', visible=True)
                with gr.Column(min_width=MIN_COL):
                    with gr.Accordion('Самочувствие (необязательно)', open=False):
                        gr.HTML('<p class="hv-small">Ответы попадут в отчёт для врача. <b>На оценку риска по фото '
                                'они не влияют.</b></p>')
                        answers = [gr.Radio(list(V.ANSWER), value='Не знаю', label=q) for _, q in WELLBEING]
                    with gr.Accordion('Для специалиста: ткань для оценки', open=False):
                        method = gr.Radio(list(V.METHOD), value='Авто', label='Ткань',
                                          info='«Авто» — как в ядре: обе ткани пригодны → модель обеих, иначе по '
                                               'пригодной. Конкретная ткань непригодна → отказ, другая не '
                                               'подставляется. «Склера» отдельно — самая слабая модель, только для '
                                               'сравнения.')
            with gr.Row(elem_classes='hv-nav'):
                back2 = gr.Button('← Назад', size='lg')
                go = gr.Button('Анализировать', variant='primary', size='lg', interactive=backend.error is None)

        # ---------------------------------------------------------------- 3. Анализ
        with gr.Column(visible=False) as analysis_scr:
            gr.HTML(V.title_html('Анализ снимка', 'Что проверено на фото и где приложение нашло ткани.'))
            with gr.Row(equal_height=False):
                with gr.Column(min_width=MIN_COL):
                    pipeline = gr.HTML(V.loading_html())
                with gr.Column(min_width=MIN_COL):
                    crop_img = gr.Image(label='Найденные ткани', type='numpy', interactive=False, visible=False,
                                        height=300)
                    face_img = gr.Image(label='Глаза на фото', type='numpy', interactive=False, visible=False,
                                        height=300)
                    legend = gr.HTML(visible=False)
            with gr.Row(elem_classes='hv-nav'):
                back3 = gr.Button('← Изменить фото или анкету', size='lg')
                to_result = gr.Button('Результат →', variant='primary', size='lg', interactive=False)

        # ---------------------------------------------------------------- 4. Результат
        with gr.Column(visible=False) as result_scr:
            gr.HTML(V.title_html('Результат скрининга'))
            with gr.Row(equal_height=False):
                with gr.Column(min_width=MIN_COL):
                    risk_card = gr.HTML()
                    hb_card = gr.HTML()
                    summary = gr.HTML()
                with gr.Column(min_width=MIN_COL):
                    quality_card = gr.HTML()
                    p_card = gr.HTML()
                    gr.Textbox(label='Вопрос к AI Personalizer', placeholder='Будет доступно в следующей версии',
                               interactive=False)
            with gr.Accordion('Для врача: технические сведения', open=False):
                doctor = gr.HTML()
            gr.HTML(DISCLAIMER_HTML)
            report = gr.DownloadButton('Скачать отчёт для врача (.txt)', visible=False, size='lg')
            with gr.Row(elem_classes='hv-nav'):
                home_btn = gr.Button('На главную', size='lg')
                again = gr.Button('Новый снимок', variant='primary', size='lg')

        gr.HTML(V.footer_html(backend))
        screens = [home, photo_scr, survey_scr, analysis_scr, result_scr, stepper]

        # ---------------------------------------------------------------- логика
        def on_next_photo(p):
            if not p:
                gr.Warning('Сначала загрузите или снимите фото.')
                return [gr.update()] * len(screens) + [V.note_html('warn', 'Сначала загрузите или снимите фото.')]
            return show('survey') + ['']

        def on_sex(s):
            return gr.update(visible=s != 'Мужской', value='Не указано')

        def on_analyze(p):
            if not p:                   # gr.Error останавливает цепочку: анализ без фото не запускается
                raise gr.Error('Нет фото: вернитесь на шаг «Фото».')
            return show('analysis') + [V.loading_html(), gr.update(visible=False), gr.update(visible=False),
                                       gr.update(visible=False), gr.update(interactive=False),
                                       gr.update(interactive=False)]

        def run(p, s, a, pr, m, w1, w2, w3):
            survey = V.parse_survey(s, a, pr, m, (w1, w2, w3))
            t0 = time.time()
            try:
                r, res = V.run_analysis(p, survey, backend)
                err = None
            except Exception as e:      # непредвиденная ошибка кода: показываем, но без какого-либо результата
                r = {'status': 'unreliable', 'stage': 'frame', 'issues': [f'ошибка программы при анализе: '
                                                                          f'{type(e).__name__}: {e}'],
                     'notes': [], 'reasons': [], 'risk': None, 'hb_range': None, 'model': None}
                res, err = None, e
            dt_s = time.time() - t0
            crop, face = V.overlays(r) if res is not None else (None, None)
            reply = personalizer.reply(make_input(r, survey))
            path = os.path.join(tempfile.mkdtemp(), 'hemavision_report.txt')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(V.report_text(r, survey, backend.version) if res is not None else
                        'Анализ не выполнен: ' + '; '.join(r['issues']) + '\n')
            return (V.pipeline_html(r, dt_s),
                    gr.update(value=crop, visible=crop is not None), gr.update(value=face, visible=face is not None),
                    gr.update(visible=crop is not None or face is not None, value=V.legend_html(crop is not None,
                                                                                               face is not None)),
                    V.risk_card_html(r) if err is None else V.error_html(err),
                    gr.update(value=V.hb_html(r), visible=bool(V.hb_html(r))),
                    V.quality_html(res) if res is not None else '', V.summary_html(r, res),
                    _personalizer_html(reply), V.doctor_html(r, backend.version), gr.update(value=path, visible=True),
                    {'r': r, 'survey': survey}, gr.update(interactive=True), gr.update(interactive=True))

        inputs = [photo, sex, age, preg, method, *answers]
        start.click(lambda: show('photo'), None, screens)
        back1.click(lambda: show('home'), None, screens)
        next1.click(on_next_photo, photo, screens + [photo_msg])
        photo.change(lambda p: '' if p else gr.update(), photo, photo_msg)
        sex.change(on_sex, sex, preg)
        back2.click(lambda: show('photo'), None, screens)
        go.click(on_analyze, photo, screens + [pipeline, crop_img, face_img, legend, to_result, go]).success(
            run, inputs, [pipeline, crop_img, face_img, legend, risk_card, hb_card, quality_card, summary, p_card, doctor,
                          report, state, to_result, go], show_progress='hidden')
        back3.click(lambda: show('survey'), None, screens)
        to_result.click(lambda: show('result'), None, screens)
        again.click(lambda: show('photo') + [None], None, screens + [photo])
        home_btn.click(lambda: show('home'), None, screens)
    return demo


def launch(model_path=None, share=False, personalizer=None, **kw):
    """Собрать и запустить. share=True — ссылка *.gradio.live (фото идут через сервер Gradio — внешний сервис)."""
    demo = build(model_path, personalizer)
    return demo.launch(share=share, **demo.hv_launch_kwargs, **kw)
