"""Тема и CSS интерфейса HemaVision: светлый медицинский стиль, белые карточки, спокойные голубые и бирюзовые акценты.

Адаптивность: телефон (< 640 px) — одна колонка, крупные элементы; планшет — одна колонка до 760 px; ноутбук и шире
(≥ 1024 px) — до 1120 px, экраны в две колонки (gr.Row + gr.Column(min_width) сами переносятся на узком экране).
Доступность: контраст текста ≥ 4,5:1 (крупного ≥ 3:1), интерактивные элементы ≥ 44 px, видимая рамка фокуса.
Шрифты системные (без загрузки из интернета: приложение задумано офлайн). Тёмная тема отключена (LIGHT_JS):
один проверенный светлый стиль.
"""

FONT = ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'Helvetica Neue', 'Arial', 'sans-serif']

# Цвета (проверены на контраст с белым и с фоном карточек; см. tests/test_hemavision_ui.py::test_palette_contrast)
INK = '#0f172a'          # основной текст
MUTED = '#475569'        # вторичный текст (7,6:1 на белом)
ACCENT = '#0e7490'       # бирюзово-голубой акцент, кнопки (5,4:1 с белым текстом)
ACCENT_DARK = '#155e75'
BG = '#f4f7fb'
LINE = '#dbe3ec'


def theme():
    import gradio as gr
    t = gr.themes.Base(primary_hue='cyan', secondary_hue='sky', neutral_hue='slate', radius_size='lg',
                       spacing_size='md', text_size='md', font=FONT)
    return t.set(
        body_background_fill=BG, body_text_color=INK, body_text_color_subdued=MUTED,
        background_fill_primary='#ffffff', background_fill_secondary=BG,
        block_background_fill='#ffffff', block_border_color=LINE, block_border_width='1px', block_shadow='none',
        block_radius='14px', block_label_background_fill='#ffffff', block_label_text_color=MUTED,
        block_label_text_size='14px', block_label_text_weight='600', block_title_text_color=INK,
        block_title_text_weight='600', block_info_text_color=MUTED,
        input_background_fill='#ffffff', input_border_color='#94a3b8', input_border_color_focus=ACCENT,
        checkbox_label_background_fill='#ffffff', checkbox_label_background_fill_selected='#ecfeff',
        checkbox_label_border_color=LINE, checkbox_label_border_color_selected=ACCENT, checkbox_label_border_width='1px',
        checkbox_label_text_color=INK, checkbox_label_text_color_selected=INK, checkbox_label_padding='10px 14px',
        checkbox_label_text_size='15px', checkbox_background_color_selected=ACCENT, checkbox_border_color_selected=ACCENT,
        button_primary_background_fill=ACCENT, button_primary_background_fill_hover=ACCENT_DARK,
        button_primary_text_color='#ffffff', button_primary_border_color=ACCENT,
        button_secondary_background_fill='#ffffff', button_secondary_background_fill_hover='#f1f5f9',
        button_secondary_text_color=INK, button_secondary_border_color='#94a3b8',
        button_large_text_size='16px', button_large_padding='12px 20px', button_large_radius='12px',
        link_text_color=ACCENT, color_accent_soft='#ecfeff', loader_color=ACCENT,
    )


# Тёмная тема Gradio включается по настройке системы; держим один светлый стиль (контраст проверен только для него)
LIGHT_JS = """() => {
  const strip = () => { document.documentElement.classList.remove('dark'); document.body.classList.remove('dark'); };
  strip(); new MutationObserver(strip).observe(document.body, {attributes: true, attributeFilter: ['class']});
}"""

CSS = """
.gradio-container {max-width: 1120px !important; margin: 0 auto !important; padding: 8px 16px 24px !important;
  box-sizing: border-box !important; width: 100% !important; min-width: 0 !important;}
@media (min-width: 641px) and (max-width: 1023px) {.gradio-container {max-width: 760px !important;}}
@media (max-width: 640px) {.gradio-container {padding: 4px 10px 20px !important;}}
.gradio-container, .gradio-container * {overflow-wrap: break-word; word-break: normal; hyphens: manual;}
.hv-card li, .hv-card p, .hv-note, .hv-note * {overflow-wrap: anywhere;}   /* очень длинные слова без пробелов в сообщениях ядра */
.gradio-container img {max-width: 100%;}
:root {
  --hv-ink: #0f172a; --hv-muted: #475569; --hv-accent: #0e7490; --hv-accent-soft: #ecfeff; --hv-line: #dbe3ec;
  --hv-low: #166534; --hv-low-bg: #f0fdf4; --hv-low-line: #86efac;
  --hv-mid: #92400e; --hv-mid-bg: #fffbeb; --hv-mid-line: #fcd34d;
  --hv-high: #b91c1c; --hv-high-bg: #fef2f2; --hv-high-line: #fca5a5;
  --hv-none: #334155; --hv-none-bg: #f1f5f9; --hv-none-line: #cbd5e1;
}
/* доступность: крупные цели, видимый фокус; вторичные кнопки с рамкой, чтобы их было видно на белом */
.gradio-container button.secondary {border: 1px solid #94a3b8 !important;}
.gradio-container button {min-height: 48px;}
.gradio-container :focus-visible {outline: 3px solid #0369a1 !important; outline-offset: 2px !important;}
.gradio-container input:focus, .gradio-container textarea:focus {outline: 3px solid #0369a1 !important;
  outline-offset: 1px !important;}
.gradio-container label:has(input[type=radio]) {min-height: 44px; align-items: center;}
.gradio-container .label-wrap {min-height: 48px; padding: 12px 4px !important; font-weight: 600; font-size: 1rem;}
.gradio-container input[type=number] {min-height: 48px; font-size: 1.05rem;}
/* мелкие кнопки самих компонентов Gradio (полноэкранный режим, удалить фото, источник фото) — до 44 px */
.gradio-container .icon-button-wrapper .icon-button, .gradio-container .source-selection button {
  min-width: 44px !important; min-height: 44px !important; width: auto !important; height: auto !important;
  padding: 10px !important;}
.gradio-container .source-selection button svg, .gradio-container .icon-button svg {width: 22px; height: 22px;}
.gradio-container textarea {min-height: 48px;}
/* типографика */
.hv-h1 {font-size: 2rem; line-height: 1.2; margin: 8px 0 6px; color: var(--hv-ink); font-weight: 700;}
.hv-h2 {font-size: 1.35rem; line-height: 1.3; margin: 4px 0 2px; color: var(--hv-ink); font-weight: 700;}
.hv-sub {color: var(--hv-muted); font-size: 1rem; margin: 0 0 6px; line-height: 1.5;}
@media (max-width: 640px) {.hv-h1 {font-size: 1.6rem;} .hv-h2 {font-size: 1.2rem;}}
.hv-card {background: #fff; border: 1px solid var(--hv-line); border-radius: 14px; padding: 16px 18px; margin: 0 0 12px;
  color: var(--hv-ink); line-height: 1.5;}
.hv-card h3 {margin: 0 0 8px; font-size: 1.05rem; font-weight: 700; color: var(--hv-ink);}
.hv-card p {margin: 6px 0; color: var(--hv-ink);}
.hv-small {font-size: .92rem; color: var(--hv-muted) !important;}
.hv-ic {display: inline-flex; flex: none; width: 1.25em; height: 1.25em; vertical-align: -0.2em;}
.hv-ic svg {width: 100%; height: 100%;}
/* главная: шаги */
.hv-steps3 {list-style: none; margin: 0; padding: 0; display: grid; gap: 10px;}
.hv-steps3 li {display: grid; grid-template-columns: 44px 1fr; gap: 12px; align-items: center;}
.hv-steps3 .num {width: 44px; height: 44px; border-radius: 50%; background: var(--hv-accent-soft); color: var(--hv-accent);
  display: flex; align-items: center; justify-content: center;}
.hv-steps3 .num .hv-ic {width: 24px; height: 24px;}
.hv-steps3 b {display: block;}
/* плашки */
.hv-note {border-radius: 12px; padding: 12px 14px; margin: 0 0 12px; font-size: .95rem; line-height: 1.5;
  display: grid; grid-template-columns: 24px 1fr; gap: 10px; align-items: start; color: var(--hv-ink);}
.hv-note .hv-ic {width: 22px; height: 22px; margin-top: 1px;}
.hv-info {background: var(--hv-accent-soft); border: 1px solid #a5f3fc;}
.hv-info .hv-ic {color: var(--hv-accent);}
.hv-warn {background: var(--hv-mid-bg); border: 1px solid var(--hv-mid-line);}
.hv-warn .hv-ic {color: var(--hv-mid);}
.hv-okline {background: var(--hv-low-bg); border: 1px solid var(--hv-low-line);}
.hv-okline .hv-ic {color: var(--hv-low);}
.hv-err {background: var(--hv-high-bg); border: 1px solid var(--hv-high-line);}
.hv-err .hv-ic {color: var(--hv-high);}
/* индикатор шагов */
.hv-stepper {display: flex; gap: 6px; list-style: none; padding: 0; margin: 4px 0 8px;}
.hv-stepper li {flex: 1 1 0; min-width: 0; white-space: nowrap; text-align: center; font-size: .9rem; color: var(--hv-muted);
  padding-top: 8px; border-top: 4px solid var(--hv-line);}
.hv-stepper li span.n {display: inline-flex; width: 26px; height: 26px; border-radius: 50%; align-items: center;
  justify-content: center; margin-right: 6px; background: #e2e8f0; color: var(--hv-ink); font-weight: 700; font-size: .85rem;}
.hv-stepper li.done, .hv-stepper li.cur {border-top-color: var(--hv-accent);}
.hv-stepper li.cur {color: var(--hv-ink); font-weight: 700;}
.hv-stepper li.cur span.n, .hv-stepper li.done span.n {background: var(--hv-accent); color: #fff;}
@media (max-width: 480px) {.hv-stepper li .t {display: none;} .hv-stepper li.cur .t {display: inline;}
  .hv-stepper li.cur {flex: 2 1 0;}}
/* инструкция съёмки */
.hv-tips {list-style: none; margin: 0; padding: 0; display: grid; gap: 12px;}
.hv-tips li {display: grid; grid-template-columns: 40px 1fr; gap: 12px; align-items: start;}
.hv-tips .ti {width: 40px; height: 40px; border-radius: 10px; background: var(--hv-accent-soft); color: var(--hv-accent);
  display: flex; align-items: center; justify-content: center;}
.hv-tips .ti .hv-ic {width: 22px; height: 22px;}
/* этапы анализа */
.hv-lead {font-weight: 700; margin: 0 0 8px;}
.hv-pipe {list-style: none; padding: 0; margin: 0;}
.hv-step {display: grid; grid-template-columns: 32px minmax(0, 1fr) auto; gap: 10px; align-items: center; padding: 10px 0;
  border-bottom: 1px solid var(--hv-line);}
.hv-step:last-child {border-bottom: 0;}
.hv-step .ic {width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center;}
.hv-step .ic .hv-ic {width: 18px; height: 18px;}
.hv-step.done .ic {background: var(--hv-low-bg); color: var(--hv-low);}
.hv-step.fail .ic {background: var(--hv-high-bg); color: var(--hv-high);}
.hv-step.skip .ic {background: var(--hv-none-bg); color: var(--hv-none);}
.hv-step .st {font-size: .88rem; color: var(--hv-muted); text-align: right;}
.hv-step.fail .st {color: var(--hv-high); font-weight: 700;}
.hv-loading {display: grid; grid-template-columns: 28px 1fr; gap: 12px; align-items: center;}
.hv-spin {width: 26px; height: 26px; border-radius: 50%; border: 3px solid #cffafe; border-top-color: var(--hv-accent);
  animation: hv-rot 1s linear infinite;}
@keyframes hv-rot {to {transform: rotate(360deg);}}
@media (prefers-reduced-motion: reduce) {.hv-spin {animation: none;}}
/* карточка риска */
.hv-risk {border-width: 2px;}
.hv-risk.low {border-color: var(--hv-low-line); background: var(--hv-low-bg);}
.hv-risk.mid {border-color: var(--hv-mid-line); background: var(--hv-mid-bg);}
.hv-risk.high {border-color: var(--hv-high-line); background: var(--hv-high-bg);}
.hv-risk.none {border-color: var(--hv-none-line); background: var(--hv-none-bg);}
.hv-badge {display: flex; align-items: center; gap: 10px; font-size: 1.4rem; font-weight: 800; letter-spacing: .01em;}
.hv-badge .hv-ic {width: 30px; height: 30px;}
.hv-risk.low .hv-badge {color: var(--hv-low);} .hv-risk.mid .hv-badge {color: var(--hv-mid);}
.hv-risk.high .hv-badge {color: var(--hv-high);} .hv-risk.none .hv-badge {color: var(--hv-none);}
.hv-explain {font-size: 1.02rem; margin: 10px 0 6px;}
.hv-issues {margin: 6px 0 8px 20px; padding: 0;}
.hv-issues li {margin: 4px 0;}
.hv-scale {display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 6px; margin-top: 12px;}
.hv-scale .seg {padding: 8px 4px; text-align: center; font-size: .9rem; border-radius: 10px; color: var(--hv-ink);
  background: #fff; border: 1px solid var(--hv-line); display: flex; flex-direction: column; gap: 2px; min-height: 44px;
  justify-content: center;}
.hv-scale .seg.on, .hv-scale .seg.on * {font-weight: 800; color: #fff !important; border-color: transparent;}
.hv-scale .seg.low.on {background: var(--hv-low);} .hv-scale .seg.mid.on {background: var(--hv-mid);}
.hv-scale .seg.high.on {background: var(--hv-high);}
.hv-scale .seg b {font-size: .78rem;}
.hv-hb {font-size: 2rem; font-weight: 800; margin: 4px 0; color: var(--hv-ink);}
.hv-hb small {font-size: 1rem; font-weight: 600; color: var(--hv-muted);}
.hv-tag {display: inline-block; font-size: .78rem; font-weight: 700; background: var(--hv-mid-bg); color: var(--hv-mid);
  border: 1px solid var(--hv-mid-line); border-radius: 999px; padding: 1px 10px; vertical-align: middle;}
.hv-qlist {list-style: none; margin: 0; padding: 0;}
.hv-qlist li {display: flex; justify-content: space-between; gap: 12px; padding: 8px 0; border-bottom: 1px solid var(--hv-line);}
.hv-qlist li:last-child {border-bottom: 0;}
.hv-qlist .v {font-weight: 700; display: inline-flex; gap: 6px; align-items: center; text-align: right;
  white-space: nowrap; overflow-wrap: normal !important;}
.hv-qlist .v.ok {color: var(--hv-low);} .hv-qlist .v.bad {color: var(--hv-high);} .hv-qlist .v.na {color: var(--hv-muted);}
.hv-doc {font-size: .92rem;} .hv-doc ul {margin: 6px 0 6px 20px; padding: 0;} .hv-doc li {margin: 3px 0;}
.hv-personalizer {border-style: dashed; border-color: #67e8f9;}
.hv-footer {text-align: center; color: var(--hv-muted); font-size: .85rem; margin-top: 8px;}
/* кнопки навигации: на телефоне — в столбик, основная сверху */
@media (max-width: 640px) {.hv-nav {flex-direction: column-reverse !important;} .hv-nav > * {width: 100% !important;}}
/* Gradio оборачивает gr.HTML в prose: убираем лишние рамки таблиц и отступы */
.hv-app .prose table, .hv-app .prose td {border: 0 !important;}
.hv-app .html-container {padding: 0 !important;}
"""
