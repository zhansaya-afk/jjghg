"""Gradio: сбор своих фото глаза + лабораторный Hb того же дня → <ANEMIA_DATA>/phone/.

    python app/collect.py --share            # локально или в Colab; ссылка *.gradio.live — открыть на телефоне
    from app.collect import build; build().launch(share=True)

Оценка риска по фото здесь не считается: только сбор (снимок, анкета, маска конъюнктивы). Вся логика — anemia.collect.
"""
import argparse
import os
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anemia import collect as C

SEX = {'Женский': 'F', 'Мужской': 'M'}
PREG = {'Не указано': None, 'Нет': False, 'Да': True}
EYE = {'Левый': 'L', 'Правый': 'R'}
MASK = {'Нарисую кистью': 'draw', 'Предложенная верна': 'auto', 'Без маски (разметят позже)': 'none'}
EDIT_SIDE = 1280              # px: размер фото в редакторе маски (маска потом растягивается до сохраняемого)


def data_root():
    return os.environ.get('ANEMIA_DATA') or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                         'data')


def _u8(rgb):
    return (np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8)


def build(root=None):
    import gradio as gr

    root = root or data_root()

    def counts():
        s = C.summary(root)
        return (f'Собрано: **{s["photos"]}** фото, **{s["patients"]}** участников '
                f'(анемия по ВОЗ у {s["anemic_patients"]}), с маской — {s["with_mask"]}. Папка: `{root}/phone`')

    def on_new_id():
        return C.new_id(C.known_ids(root))

    def on_check(path):
        if not path:
            return None, None, '⚠️ Загрузите или снимите фото.', list(MASK)[0]
        try:
            rgb = C.load_for_collect(path)
        except ValueError as e:                   # PhotoError: не изображение, ч/б, слишком мало
            return None, None, f'❌ {e}', list(MASK)[0]
        q = C.check_photo(rgb)
        small = C.resize_max(rgb, EDIT_SIDE)
        conj = C.resize_to(q['conj'], small.shape[:2]) if q['conj'].any() else np.zeros(small.shape[:2], bool)
        # предложение — только контур на фоне: предзаполненный слой Gradio 6 не стирается ластиком надёжно
        bg = C.outline(_u8(small), conj) if conj.any() else _u8(small)
        editor = {'background': bg, 'layers': [], 'composite': bg}
        lines = [('✅ Снимок годен' if q['ok'] else '❌ Переснимите: ' + '; '.join(q['blocking'])) +
                 (f' (резкость {q["blur"]:.2f}, норма ≥ 1,3)' if not np.isnan(q['blur']) else '')]
        lines += ['⚠️ ' + w for w in q['warnings']]
        if q['conj'].any():
            lines.append('Голубой контур — предложенная маска. Верна → «Предложенная верна» (кистью можно дорисовать '
                         'пропущенное). Неверна → закрасьте конъюнктиву кистью и выберите «Нарисую кистью».')
        state = {'rgb': rgb, 'ok': q['ok'], 'blur': q['blur'], 'auto': q['conj'] if q['conj'].any() else None}
        choice = list(MASK)[1] if q['conj'].any() else list(MASK)[0]
        return state, editor, '\n\n'.join(lines), choice

    def on_save(state, editor, consent, pid, hb, unit, sex, age, preg, eye, gap, mask_choice):
        if not state:
            return '❌ Сначала нажмите «Проверить снимок».', counts()
        if not state['ok']:
            return '❌ Снимок не прошёл проверку — переснимите.', counts()
        try:
            meta = C.subject_meta(pid, hb, unit, SEX.get(sex), age, PREG.get(preg), EYE.get(eye), gap, consent)
            h, w = state['rgb'].shape[:2]
            drawn = C.mask_from_layers((editor or {}).get('layers'), (h, w))
            mask, src = C.choose_mask(drawn, state['auto'], MASK.get(mask_choice))
            out = C.save_record(root, state['rgb'], meta, mask, src, state['blur'])
        except C.CollectError as e:
            return f'❌ {e}', counts()
        return f'✅ Сохранено: `{out["file"]}`' + (' + маска' if out['mask'] else ' (без маски)'), counts()

    def on_zip():
        d = os.path.join(root, 'phone')
        if not os.path.isdir(d):
            return None
        base = os.path.join(tempfile.mkdtemp(), 'phone_' + time.strftime('%Y%m%d-%H%M'))
        return shutil.make_archive(base, 'zip', d)

    with gr.Blocks(title='Сбор фото + Hb') as demo:
        gr.Markdown('## Сбор фото конъюнктивы с анализом крови\nТолько сбор данных: оценка риска здесь не '
                    'выдаётся. Никаких имён — только сгенерированный id.')
        stats = gr.Markdown(counts())
        state = gr.State(None)
        with gr.Row():
            with gr.Column():
                gr.Markdown('### 1. Участник')
                consent = gr.Checkbox(label='Письменное согласие участника получено', value=False)
                with gr.Row():
                    pid = gr.Textbox(label='id участника (не имя!)', placeholder='P3fa9c1', scale=3)
                    gr.Button('Новый участник', scale=1).click(on_new_id, None, pid)
                with gr.Row():
                    hb = gr.Textbox(label='Hb из анализа', placeholder='125')
                    unit = gr.Dropdown(['г/л', 'г/дл', 'ммоль/л'], value='г/л', label='Единица')
                with gr.Row():
                    sex = gr.Radio(list(SEX), label='Пол')
                    age = gr.Number(label='Возраст, лет', precision=1)
                preg = gr.Radio(list(PREG), value='Не указано', label='Беременность')
                gap = gr.Radio([0, 1], value=0, label='Дней между анализом крови и фото')
                eye = gr.Radio(list(EYE), label='Глаз')
            with gr.Column():
                gr.Markdown('### 2. Снимок')
                gr.Markdown(C.PROTOCOL)
                photo = gr.Image(type='filepath', sources=['upload', 'webcam'], label='Фото глаза (оттянутое веко)')
                check = gr.Button('Проверить снимок', variant='secondary')
                report = gr.Markdown()
        gr.Markdown('### 3. Маска конъюнктивы — кисть (голубая) закрашивает розовую слизистую века, ластик стирает лишнее')
        editor = gr.ImageEditor(type='numpy', format='png', transforms=(), sources=(), layers=False,
                                brush=gr.Brush(colors=['#00e5ff'], color_mode='fixed', default_size=12),
                                eraser=gr.Eraser(default_size=12), label='Маска')
        mask_choice = gr.Radio(list(MASK), value=list(MASK)[0], label='Какую маску сохранить')
        with gr.Row():
            save = gr.Button('Сохранить запись', variant='primary')
            zbtn = gr.Button('Скачать архив data/phone')
        status = gr.Markdown()
        zfile = gr.File(label='Архив (скачайте: в Colab папка стирается вместе со средой)')
        check.click(on_check, photo, [state, editor, report, mask_choice])
        save.click(on_save, [state, editor, consent, pid, hb, unit, sex, age, preg, eye, gap, mask_choice],
                   [status, stats])
        zbtn.click(on_zip, None, zfile)
    return demo


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--data', help='корень данных (по умолчанию ANEMIA_DATA или ./data)')
    ap.add_argument('--share', action='store_true', help='публичная ссылка *.gradio.live (нужно в Colab)')
    a = ap.parse_args()
    build(a.data).launch(share=a.share)


if __name__ == '__main__':
    main()
