---
name: train-segmentation
description: Обучение сегментации конъюнктивы (и склеры) — U-Net/DeepLabV3+ из segmentation_models_pytorch на лёгком энкодере по маскам врачей, аугментации света, IoU/Dice на невиданных пациентах, сравнение с поиском по правилам, сохранение в models/.
---

# train-segmentation

Код — `scripts/train_seg.py` + `anemia/segment.py` (нет — создать). Обучение на GPU — в Colab: написать команде, какие ячейки запустить.

1. **Данные**: маски врачей Eyes-defy (+ наши из `data/phone/`); фолды манифеста по пациенту. Склера — только при наличии масок.
2. **Модель**: `smp.Unet` или `smp.DeepLabV3Plus`, энкодер `mobilenet_v2` / `timm-mobilenetv3_large_100` / `efficientnet-b0`
   (ImageNet), вход 256–384 px, loss Dice + BCE.
3. **Аугментации** (только train): цветовая температура 2700–7500 K (`anemia/augment.py`), экспозиция, гамма, JPEG, шум,
   размытие, масштаб, поворот; геометрия — одинаково для фото и маски (маска — nearest).
4. **Оценка**: IoU/Dice на тестовом фолде (невиданные пациенты), медиана и 95% ДИ; пустые маски — явно.
   Цель IoU ≥ 0,75 (база 0,33).
5. **Сравнение** на тех же фолдах: сеть vs `anemia/regions.py` (правила) vs `anemia/segment_baseline.py` (v11) —
   через `scripts/evaluate.py --task seg`, затем `/evaluate-model` для Hb на новых масках.
6. **Сохранение**: `models/seg_<encoder>_<дата>.pt` + JSON с энкодером, размером входа, фолдом, IoU. `models/` — не в git;
   команда скачивает файл из Colab.
7. Ранняя остановка и выбор эпохи — по валидации внутри обучающих фолдов, не по тесту.
