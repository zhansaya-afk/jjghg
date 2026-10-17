# Скрининг риска анемии по фото глаза (PoC)

Не диагноз: шкала риска, ориентировочный диапазон Hb и отчёт для врача по фото нижней конъюнктивы и склеры.

- Прототип v11: [notebooks/anemia_screening_v11.ipynb](notebooks/anemia_screening_v11.ipynb) — [Open in Colab](https://colab.research.google.com/github/zhansaya-afk/jjghg/blob/main/notebooks/anemia_screening_v11.ipynb). Перед запуском перетащите `cp_anemic_features.npz` в панель «Файлы».
- Обучение и проверка (тонкий блокнот, код берётся из `main`): [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb) — [Open in Colab](https://colab.research.google.com/github/zhansaya-afk/jjghg/blob/main/notebooks/colab_train.ipynb).
- Прототип v12: [notebooks/anemia_screening_v12.ipynb](notebooks/anemia_screening_v12.ipynb) — [Open in Colab](https://colab.research.google.com/github/zhansaya-afk/jjghg/blob/main/notebooks/anemia_screening_v12.ipynb).
- Тесты: `pip install -e ".[dev]"` и `python -m pytest -q`; на каждый PR и push в `main` их запускает GitHub Actions (`.github/workflows/tests.yml`).
- Правила разработки и план: [CLAUDE.md](CLAUDE.md).

Данные и модели в репозиторий не коммитятся (`data/`, `models/`).
