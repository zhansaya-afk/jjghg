"""HemaVision: интерфейс скрининга риска анемии по фото глаза (Gradio). Не диагноз.

    python -m app.hemavision            # локально; --share — публичная ссылка *.gradio.live
    from app.hemavision import launch; launch(share=False)

UI ничего не считает сам: анализ — anemia.predict.analyze_photo (та же модель Я1, QC, зоны риска, выбор ткани),
отображение — app.hemavision.view (чистые функции, без gradio), AI Personalizer — app.hemavision.personalizer
(пока заглушка с готовым интерфейсом).
"""


def build(*args, **kw):
    from .ui import build as _build
    return _build(*args, **kw)


def launch(*args, **kw):
    from .ui import launch as _launch
    return _launch(*args, **kw)
