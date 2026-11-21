# Эксперименты HemaVision — реестр

| Что | Коммит | Статус |
|---|---|---|
| **Baseline (основная модель)** | `main` `0d4e2db` | текущий baseline, см. [BASELINE.md](BASELINE.md) |
| Ансамбль 5 seed (PR #41) | `fa3e7ee` на `claude/project-thread-8ktljc` | **экспериментальный кандидат; улучшение не доказано (ΔAUC +0,002 [−0,006; +0,010]); в основную модель не принят**; PR открыт, не сливать до проверки на данных Казахстана — [2026-11-21_core_seeds/](2026-11-21_core_seeds/README.md) |

Правила: кандидат принимается в `main` только при доказанном улучшении на тех же фолдах по пациенту и после решения
команды. Агрегаты каждого эксперимента — строка в `results/experiments.csv`; пациентские строки — не в репозиторий.

## Архивы Colab — сохранить вне Colab (в репозиторий не класть: строки по пациентам)
| Архив | Что внутри | Где сейчас |
|---|---|---|
| `hemavision_eval_0d4e2db.zip` | baseline `0d4e2db`: запуски core, country, seg_ext — OOF, run.json, metrics_history, логи | `/content/hv_eval_0d4e2db/` в Colab пользователя |
| `hemavision_core_seeds.zip` | эксперимент ансамбля: summary.csv, paired.csv, oof_*.csv, decision.md, run.json, лог | `/content/hv_eval_970ef2e/` в Colab пользователя |

`/content` стирается при удалении рантайма — до сохранения архивов рантайм не удалять и не перезапускать.
