# Скрининг риска анемии по фото глаза (PoC)

Не диагноз: шкала риска, ориентировочный диапазон Hb и отчёт для врача по фото нижней конъюнктивы и склеры.

- Прототип v11: [notebooks/anemia_screening_v11.ipynb](notebooks/anemia_screening_v11.ipynb) — [Open in Colab](https://colab.research.google.com/github/zhansaya-afk/jjghg/blob/main/notebooks/anemia_screening_v11.ipynb). Архив: его детская часть (CP-AnemiC, `cp_anemic_features.npz`) устарела — датасет исключён 2026-10-19.
- Обучение и проверка (тонкий блокнот, код берётся из `main`): [notebooks/colab_train.ipynb](notebooks/colab_train.ipynb) — [Open in Colab](https://colab.research.google.com/github/zhansaya-afk/jjghg/blob/main/notebooks/colab_train.ipynb).
- Прототип v12: [notebooks/anemia_screening_v12.ipynb](notebooks/anemia_screening_v12.ipynb) — [Open in Colab](https://colab.research.google.com/github/zhansaya-afk/jjghg/blob/main/notebooks/anemia_screening_v12.ipynb).
- Тесты: `pip install -e ".[dev]"` и `python -m pytest -q`; на каждый PR и push в `main` их запускает GitHub Actions (`.github/workflows/tests.yml`).
- Правила разработки и план: [CLAUDE.md](CLAUDE.md).

Данные и модели в репозиторий не коммитятся (`data/`, `models/`).

## API для приложения (`anemia/api.py`)

```python
from anemia.api import load_models, screen
models = load_models('models/hemavision_core.joblib')   # один раз при старте (модель — ячейка Я1 / scripts/train_core.py)
res = screen(jpeg_bytes, sex='F', age=30, pregnant=None, symptoms=['слабость'], models=models)
res.to_json()                                            # маски и вырезка — PNG в base64; to_json(images=False) — без них
```

- Вход: `image` — bytes / `numpy` H×W×3 / `PIL.Image` (пути к файлам не принимаются); `sex` — `M`/`F`/`М`/`Ж`/`None`; `age` — число лет 0–120 или `None`; `pregnant` — `True`/`False`/`None`; `symptoms` — список или словарь. Неверный ввод → `TypeError`/`ValueError`.
- `Result`: `risk` (низкий / средний / высокий / не определён), `p_anemia`, `hb_interval_g_l` (только если модель Hb на CV лучше константы, иначе `None` + `hb_note`), `tissue` (конъюнктива / склера / обе) и `reason`, `quality` (`sharpness`, `light`, `tissue_visible`), `warnings`, `masks` + `crop`, `model_version`, `status`, `stage`.
- Плохой снимок, не глаз, ребёнок младше 5 лет, нетипичный снимок (OOD) → `risk="не определён"` и причина в `reason`, без исключения и без цифр.
- Внутри — `anemia.predict.analyze_photo` без изменений. `quality` выводится из сообщений существующих проверок (новых измерений нет). Симптомы проверяются и возвращаются, но на риск пока не влияют. Анализ крови — отдельно: `anemia.labs.lab_hb_report`.
