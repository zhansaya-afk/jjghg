"""Раскладка интерфейса HemaVision (Gradio 6): мастер из 5 экранов.

Главная → Фото → Анкета → Анализ → Результат (+ блок AI Personalizer). Экраны — колонки, видна одна;
логика отображения — view.py, персонализатор — personalizer.py. Добавить экран: колонка в SCREENS + кнопки.
"""
import inspect
import os
import tempfile
import time

from . import view as V
from .personalizer import WELLBEING, Personalizer, StubPersonalizer, make_input
from .theme import CSS, theme

assert len(WELLBEING) == 3, 'run() принимает 3 ответа о самочувствии — поправьте сигнатуру'
SCREENS = ('home', 'photo', 'survey', 'analysis', 'result')
STEP_OF = {'home': 0, 'photo': 1, 'survey': 2, 'analysis': 3, 'result': 4}
DISCLAIMER_HTML = ('<div class="hv-note hv-disclaimer"><b>HemaVision — предварительный скрининг, не диагностика.</b> '
                   'Приложение оценивает риск анемии по цвету слизистой века и белка глаза. Диагноз ставит врач '
                   'по общему анализу крови.</div>')
HOME_HTML = """
<div class="hv-hero"><h1>HemaVision</h1>
<p>Скрининг риска анемии по фото глаза. Фото обрабатывается в этой среде, без облачных сервисов анализа, и не
сохраняется в данные проекта.</p></div>
<div class="hv-grid">
  <div class="hv-card"><b>1</b>Сфотографируйте глаз с оттянутым нижним веком</div>
  <div class="hv-card"><b>2</b>Ответьте на короткую анкету: пол, возраст</div>
  <div class="hv-card"><b>3</b>Получите шкалу риска и отчёт для врача</div>
</div>
"""
PHOTO_TIPS = """**Как снимать**
- Дневной свет у окна, без прямого солнца, **без вспышки** и фильтров камеры.
- Оттяните нижнее веко пальцем, смотрите вверх: розовая слизистая должна быть видна полосой.
- Телефон в 10–15 см от глаза, фокус тапом по веку.
- Загружайте **оригинал из камеры**: не скриншот и не фото из мессенджера."""


def _blocks_kwargs(gr):
    """Gradio 6 принимает theme/css в launch(), Gradio 5 — в Blocks(). Возвращает (для Blocks, для launch)."""
    style = {'theme': theme(), 'css': CSS}
    if 'css' in inspect.signature(gr.Blocks.__init__).parameters:
        return style, {}
    return {}, style


def build(model_path=None, personalizer: Personalizer | None = None, backend=None):
    """→ gr.Blocks. backend — готовый view.Backend (для тестов); иначе модель Я1 грузится здесь один раз."""
    import gradio as gr

    backend = backend if backend is not None else V.load_backend(model_path)
    personalizer = personalizer or StubPersonalizer()
    blocks_kw, launch_kw = _blocks_kwargs(gr)

    def show(name):
        return [gr.update(visible=s == name) for s in SCREENS] + [V.stepper_html(STEP_OF[name])]

    with gr.Blocks(title='HemaVision — скрининг риска анемии', **blocks_kw) as demo:
        demo.hv_launch_kwargs = launch_kw
        state = gr.State(None)            # {'r': dict analyze_photo, 'survey': Survey}
        stepper = gr.HTML('')

        # ---------------------------------------------------------------- 0. Главная
        with gr.Column(visible=True) as home:
            gr.HTML(HOME_HTML)
            gr.HTML(DISCLAIMER_HTML)
            gr.HTML(V.model_status_html(backend))
            start = gr.Button('Начать проверку', variant='primary', size='lg', elem_classes='hv-nav')

        # ---------------------------------------------------------------- 1. Фото
        with gr.Column(visible=False) as photo_scr:
            gr.Markdown('### Фото глаза')
            photo = gr.Image(type='filepath', image_mode=None, sources=['upload', 'webcam'],
                             label='Загрузите фото или снимите камерой', height=360)
            with gr.Accordion('Как снимать', open=True):
                gr.Markdown(PHOTO_TIPS)
            photo_msg = gr.Markdown()
            with gr.Row(elem_classes='hv-nav'):
                back1 = gr.Button('← Назад')
                next1 = gr.Button('Далее: анкета →', variant='primary')

        # ---------------------------------------------------------------- 2. Анкета
        with gr.Column(visible=False) as survey_scr:
            gr.Markdown('### Анкета')
            sex = gr.Radio(list(V.SEX), value='Женский', label='Пол')
            age = gr.Number(label='Возраст, полных лет', value=None, minimum=0, maximum=120, precision=0)
            preg = gr.Radio(list(V.PREG), value='Не указано', label='Беременность', visible=True)
            gr.Markdown('<small>Пол, возраст и беременность нужны для нормы гемоглобина ВОЗ. Детям младше 5 лет '
                        'оценка по фото не проводится.</small>')
            with gr.Accordion('Самочувствие (необязательно)', open=False):
                gr.Markdown('<small>Ответы попадут в отчёт для врача и в будущий AI Personalizer. '
                            '**На оценку риска по фото они не влияют.**</small>')
                answers = [gr.Radio(list(V.ANSWER), value='Не знаю', label=q) for _, q in WELLBEING]
            with gr.Accordion('Для специалиста: ткань для оценки', open=False):
                method = gr.Radio(list(V.METHOD), value='Авто', label='Ткань',
                                  info='«Авто» — как в ядре: обе ткани пригодны → модель обеих, иначе по пригодной. '
                                       'Конкретная ткань непригодна → отказ, другая не подставляется. '
                                       '«Склера» отдельно — самая слабая модель, только для сравнения.')
            with gr.Row(elem_classes='hv-nav'):
                back2 = gr.Button('← Назад')
                go = gr.Button('Анализировать', variant='primary', interactive=backend.error is None)

        # ---------------------------------------------------------------- 3. Анализ
        with gr.Column(visible=False) as analysis_scr:
            gr.Markdown('### Анализ снимка')
            pipeline = gr.HTML('<div class="hv-card">Идёт анализ…</div>')
            crop_img = gr.Image(label='Найденные ткани: зелёный — конъюнктива, голубой — склера', type='numpy',
                                interactive=False, visible=False, height=300)
            face_img = gr.Image(label='Глаза на фото: зелёная рамка — принят, красная — отказ', type='numpy',
                                interactive=False, visible=False, height=300)
            with gr.Row(elem_classes='hv-nav'):
                back3 = gr.Button('← Изменить фото или анкету')
                to_result = gr.Button('Результат →', variant='primary')

        # ---------------------------------------------------------------- 4. Результат
        with gr.Column(visible=False) as result_scr:
            gr.Markdown('### Результат скрининга')
            risk_card = gr.HTML()
            hb_card = gr.HTML()
            quality_card = gr.HTML()
            with gr.Accordion('Почему такой ответ', open=False):
                reasons = gr.Markdown()
            with gr.Group(elem_classes='hv-personalizer'):
                p_title = gr.Markdown()
                p_text = gr.Markdown()
                gr.Textbox(label='Вопрос к AI Personalizer', placeholder='Будет доступно в следующей версии',
                           interactive=False)
            gr.HTML(DISCLAIMER_HTML)
            report = gr.DownloadButton('Скачать отчёт для врача (.txt)', visible=False)
            with gr.Row(elem_classes='hv-nav'):
                again = gr.Button('Новый снимок', variant='primary')
                home_btn = gr.Button('На главную')

        screens = [home, photo_scr, survey_scr, analysis_scr, result_scr, stepper]

        # ---------------------------------------------------------------- логика
        def on_next_photo(p):
            if not p:
                return [gr.update()] * len(screens) + ['⚠️ Сначала загрузите или снимите фото.']
            return show('survey') + ['']

        def on_sex(s):
            return gr.update(visible=s != 'Мужской', value='Не указано')

        def on_analyze(p):
            if not p:                   # gr.Error останавливает цепочку: анализ без фото не запускается
                raise gr.Error('Нет фото: вернитесь на шаг «Фото».')
            return show('analysis')

        def run(p, s, a, pr, m, w1, w2, w3, progress=gr.Progress()):
            survey = V.parse_survey(s, a, pr, m, (w1, w2, w3))
            progress(0.2, desc='Поиск глаза, сегментация, проверка качества, модель…')
            t0 = time.time()
            try:
                r, res = V.run_analysis(p, survey, backend)
            except Exception as e:      # непредвиденная ошибка кода: показываем, но без какого-либо результата
                r = {'status': 'unreliable', 'stage': 'frame', 'issues': [f'ошибка анализа: {type(e).__name__}: {e}'],
                     'notes': [], 'reasons': [], 'risk': None, 'hb_range': None, 'model': None}
                res = None
            dt_s = time.time() - t0
            crop, face = V.overlays(r) if res is not None else (None, None)
            reply = personalizer.reply(make_input(r, survey))
            path = os.path.join(tempfile.mkdtemp(), 'hemavision_report.txt')
            with open(path, 'w', encoding='utf-8') as f:
                f.write(V.report_text(r, survey, backend.version) if res is not None else
                        'Анализ не выполнен: ' + '; '.join(r['issues']) + '\n')
            return (V.pipeline_html(r) + f'<p class="hv-small">Время анализа: {dt_s:.1f} с</p>',
                    gr.update(value=crop, visible=crop is not None), gr.update(value=face, visible=face is not None),
                    V.risk_card_html(r), V.hb_html(r), V.quality_html(res) if res is not None else '',
                    V.reasons_md(r), f'#### {reply.title}', reply.text, gr.update(value=path, visible=True),
                    {'r': r, 'survey': survey})

        inputs = [photo, sex, age, preg, method, *answers]
        start.click(lambda: show('photo'), None, screens)
        back1.click(lambda: show('home'), None, screens)
        next1.click(on_next_photo, photo, screens + [photo_msg])
        sex.change(on_sex, sex, preg)
        back2.click(lambda: show('photo'), None, screens)
        go.click(on_analyze, photo, screens).success(
            run, inputs, [pipeline, crop_img, face_img, risk_card, hb_card, quality_card, reasons, p_title, p_text,
                          report, state])
        back3.click(lambda: show('survey'), None, screens)
        to_result.click(lambda: show('result'), None, screens)
        again.click(lambda: show('photo') + [None], None, screens + [photo])
        home_btn.click(lambda: show('home'), None, screens)
    return demo


def launch(model_path=None, share=False, personalizer=None, **kw):
    """Собрать и запустить. share=True — ссылка *.gradio.live (фото идут через сервер Gradio — внешний сервис)."""
    demo = build(model_path, personalizer)
    return demo.launch(share=share, **demo.hv_launch_kwargs, **kw)
