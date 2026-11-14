import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app.hemavision.ui import launch


def main():
    ap = argparse.ArgumentParser(description='HemaVision: интерфейс скрининга риска анемии по фото (Gradio)')
    ap.add_argument('--model', help='файл модели ядра (по умолчанию $ANEMIA_MODELS/hemavision_core.joblib)')
    ap.add_argument('--share', action='store_true', help='публичная ссылка *.gradio.live (фото идут через сервер Gradio)')
    a = ap.parse_args()
    launch(model_path=a.model, share=a.share)


if __name__ == '__main__':
    main()
