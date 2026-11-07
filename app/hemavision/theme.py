"""Тема и CSS интерфейса HemaVision: спокойный медицинский стиль, одна колонка до 760 px (телефон и ПК).
Шрифты системные (без загрузки из интернета: приложение задумано офлайн)."""

FONT = ['Inter', 'system-ui', '-apple-system', 'Segoe UI', 'Roboto', 'sans-serif']


def theme():
    import gradio as gr
    return gr.themes.Soft(primary_hue='teal', secondary_hue='sky', neutral_hue='slate', radius_size='lg',
                          font=FONT)


CSS = """
.gradio-container {max-width: 760px !important; margin: 0 auto !important; padding: 0 12px !important;}
:root {
  --hv-low: #15803d; --hv-low-bg: #ecfdf3; --hv-mid: #b45309; --hv-mid-bg: #fffbeb;
  --hv-high: #b91c1c; --hv-high-bg: #fef2f2; --hv-none: #475569; --hv-none-bg: #f1f5f9;
  --hv-card: #ffffff; --hv-line: #e2e8f0; --hv-text: #0f172a; --hv-muted: #475569; --hv-accent: #0f766e;
}
.dark {
  --hv-low: #4ade80; --hv-low-bg: #052e16; --hv-mid: #fbbf24; --hv-mid-bg: #3a2604; --hv-high: #f87171;
  --hv-high-bg: #3b0d0d; --hv-none: #cbd5e1; --hv-none-bg: #1e293b; --hv-card: #0f172a; --hv-line: #334155;
  --hv-text: #f1f5f9; --hv-muted: #cbd5e1; --hv-accent: #2dd4bf;
}
.hv-hero {padding: 20px 4px 4px;}
.hv-hero h1 {font-size: 1.9rem; margin: 0 0 4px; color: var(--hv-accent);}
.hv-hero p {color: var(--hv-muted); margin: 0;}
.hv-card {background: var(--hv-card); border: 1px solid var(--hv-line); border-radius: 14px; padding: 16px;
  margin: 8px 0; color: var(--hv-text);}
.hv-card h3 {margin: 0 0 8px; font-size: 1.05rem;}
.hv-grid {display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;}
.hv-grid .hv-card {margin: 0;}
.hv-grid b {display: block; color: var(--hv-accent); font-size: 1.4rem;}
@media (max-width: 600px) {.hv-grid {grid-template-columns: 1fr;} .hv-hero h1 {font-size: 1.5rem;}}
.hv-note {border-radius: 12px; padding: 10px 14px; margin: 8px 0; font-size: .92rem; border: 1px solid var(--hv-line);}
.hv-disclaimer {background: var(--hv-none-bg); color: var(--hv-text); border-left: 4px solid var(--hv-accent);}
.hv-warn {background: var(--hv-mid-bg); color: var(--hv-text); border-left: 4px solid var(--hv-mid);}
.hv-okline {background: var(--hv-low-bg); color: var(--hv-text); border-left: 4px solid var(--hv-low);}
.hv-stepper {display: flex; gap: 6px; list-style: none; padding: 0; margin: 12px 0 4px; counter-reset: s;}
.hv-stepper li {flex: 1; text-align: center; font-size: .82rem; color: var(--hv-muted); padding-top: 4px;
  border-top: 4px solid var(--hv-line);}
.hv-stepper li span {display: inline-block; width: 1.4em; height: 1.4em; line-height: 1.4em; border-radius: 50%;
  margin-right: 4px; background: var(--hv-line); color: var(--hv-text); font-weight: 600;}
.hv-stepper li.done {border-top-color: var(--hv-accent);}
.hv-stepper li.cur {border-top-color: var(--hv-accent); color: var(--hv-text); font-weight: 600;}
.hv-stepper li.cur span, .hv-stepper li.done span {background: var(--hv-accent); color: #fff;}
.hv-lead {font-weight: 600; margin: 0 0 8px;}
.hv-steps {list-style: none; padding: 0; margin: 0;}
.hv-step {display: grid; grid-template-columns: 2em 1fr auto; gap: 8px; align-items: center; padding: 7px 0;
  border-bottom: 1px solid var(--hv-line);}
.hv-step:last-child {border-bottom: 0;}
.hv-step .ic {font-weight: 700; text-align: center; border-radius: 50%; width: 1.6em; height: 1.6em; line-height: 1.6em;}
.hv-step.done .ic {background: var(--hv-low-bg); color: var(--hv-low);}
.hv-step.fail .ic {background: var(--hv-high-bg); color: var(--hv-high);}
.hv-step.skip .ic {background: var(--hv-none-bg); color: var(--hv-none);}
.hv-step .st {font-size: .82rem; color: var(--hv-muted);}
.hv-step.fail .st {color: var(--hv-high); font-weight: 600;}
.hv-risk {border-width: 2px;}
.hv-risk.low {border-color: var(--hv-low); background: var(--hv-low-bg);}
.hv-risk.mid {border-color: var(--hv-mid); background: var(--hv-mid-bg);}
.hv-risk.high {border-color: var(--hv-high); background: var(--hv-high-bg);}
.hv-risk.none {border-color: var(--hv-none); background: var(--hv-none-bg);}
.hv-badge {display: flex; align-items: center; gap: 10px; font-size: 1.35rem; font-weight: 800; letter-spacing: .02em;}
.hv-badge .ic {display: inline-flex; align-items: center; justify-content: center; min-width: 1.8em; height: 1.8em;
  border-radius: 50%; color: #fff; font-size: 1rem;}
.hv-risk.low .hv-badge {color: var(--hv-low);} .hv-risk.low .hv-badge .ic {background: var(--hv-low);}
.hv-risk.mid .hv-badge {color: var(--hv-mid);} .hv-risk.mid .hv-badge .ic {background: var(--hv-mid);}
.hv-risk.high .hv-badge {color: var(--hv-high);} .hv-risk.high .hv-badge .ic {background: var(--hv-high);}
.hv-risk.none .hv-badge {color: var(--hv-none);} .hv-risk.none .hv-badge .ic {background: var(--hv-none);}
.hv-explain {margin: 10px 0 6px; font-size: 1rem;}
.hv-issues {margin: 4px 0 8px 18px; padding: 0;}
.hv-scale {display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; margin-top: 12px;}
.hv-scale .seg {padding: 8px 4px 6px; text-align: center; font-size: .85rem; border-radius: 8px; opacity: .45;
  color: #fff; display: flex; flex-direction: column; gap: 2px;}
.hv-scale .seg.low {background: var(--hv-low);} .hv-scale .seg.mid {background: var(--hv-mid);}
.hv-scale .seg.high {background: var(--hv-high);}
.hv-scale .seg.on {opacity: 1; outline: 3px solid var(--hv-text); outline-offset: 1px; font-weight: 700;}
.hv-scale .seg b {font-size: .75rem;}
.hv-hb {font-size: 2rem; font-weight: 800; margin: 4px 0; color: var(--hv-text);}
.hv-hb small {font-size: 1rem; font-weight: 500; color: var(--hv-muted);}
.hv-tag {font-size: .72rem; font-weight: 600; background: var(--hv-mid-bg); color: var(--hv-mid);
  border: 1px solid var(--hv-mid); border-radius: 999px; padding: 1px 8px; vertical-align: middle;}
.hv-small {font-size: .88rem; color: var(--hv-muted); margin: 4px 0 0;}
.hv-table {width: 100%; border-collapse: collapse;}
.hv-table td {padding: 6px 0; border-bottom: 1px solid var(--hv-line);}
.hv-table td:last-child {text-align: right; font-weight: 600;}
.hv-personalizer {border: 1px dashed var(--hv-accent) !important; border-radius: 14px !important;}
.hv-nav button {min-height: 48px;}
"""
