"""Стабильность режима «несколько фото» на симулированных сериях (Eyes-defy). Цифры точности не считает.

У каждого пациента из исходного снимка делаются две независимые «серии» по --k фото: другой свет
(augment.random_relight — T 2700–7500 K, экспозиция ×0,6–1,6, гамма), небольшой поворот/масштаб и пересжатие JPEG
(так меняется и маска века, как при пересъёмке). Метрика — как часто ответ меняется между двумя попытками:
  один снимок:  уровень(серия A, фото 1) ≠ уровень(серия B, фото 1);
  серия:        уровень(медиана A) ≠ уровень(медиана B).
Плюс медиана |ΔP| между попытками и доля фото/серий, прошедших QC.

Ограничения (говорить вместе с цифрами): модель из Я1 обучена на ВСЕХ пациентах Eyes-defy, в том числе на этих, и
на копиях со светом random_relight → цифры оптимистичны; настоящая серия — это разные снимки (поза, веко, блики),
симуляция их не воспроизводит; польза на точности не измеряется (один снимок на пациента).

    python scripts/series_stability.py --data data --n 40 --k 5      # ~10–20 мин в Colab
"""
import argparse
import io
import json
import os
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from anemia import predict as PR
from anemia import series as SR
from anemia.augment import random_relight


def perturb(rgb, rng):
    """Один «повторный снимок»: свет + поворот ±5° + масштаб 0,9–1,1 + JPEG 80–95. rgb float 0–1 → байты JPEG."""
    out = random_relight(rgb, rng)
    im = Image.fromarray((np.clip(out, 0, 1) * 255).astype(np.uint8))
    s = rng.uniform(0.9, 1.1)
    im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
    im = im.rotate(rng.uniform(-5, 5), resample=Image.BILINEAR, expand=False, fillcolor=(0, 0, 0))
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=int(rng.integers(80, 96)))
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('ANEMIA_DATA', 'data'))
    ap.add_argument('--model', default=None, help='файл модели (по умолчанию как в Я2)')
    ap.add_argument('--n', type=int, default=40, help='пациентов (случайная выборка, seed)')
    ap.add_argument('--k', type=int, default=5, help='фото в серии')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    man = pd.read_csv(os.path.join(a.data, 'manifest.csv'))
    man = man[man.dataset == 'eyes_defy'].drop_duplicates('group')
    man = man.sample(min(a.n, len(man)), random_state=a.seed)
    bundle = PR.load_bundle(a.model)
    lm, _ = PR.face_landmarker()
    rng = np.random.default_rng(a.seed)
    rows, t0 = [], time.time()
    for j, m in enumerate(man.itertuples(), 1):
        rgb = PR.load_photo(os.path.join(a.data, m.image))
        age = None if pd.isna(m.age) else float(m.age)
        res = []
        for _ in range(2):
            photos = [perturb(rgb, rng) for _ in range(a.k)]
            res.append(SR.analyze_series(photos, sex=m.sex, age=age, bundle=bundle, landmarker=lm, min_ok=2))
        A, B = res
        a1, b1 = A['results'][0], B['results'][0]
        rows.append({
            'group': m.group, 'anemic': bool(m.anemic),
            'single_ok': a1['status'] == 'ok' and b1['status'] == 'ok',
            'single_flip': a1['risk'] != b1['risk'] if a1['status'] == b1['status'] == 'ok' else np.nan,
            'single_dp': abs(a1['p_anemia'] - b1['p_anemia']) if a1['status'] == b1['status'] == 'ok' else np.nan,
            'series_ok': A['status'] == 'ok' and B['status'] == 'ok',
            'series_flip': A['risk'] != B['risk'] if A['status'] == B['status'] == 'ok' else np.nan,
            'series_dp': abs(A['p_anemia'] - B['p_anemia']) if A['status'] == B['status'] == 'ok' else np.nan,
            'photos_ok': (A['n_used'] + B['n_used']) / (2 * a.k),
            'levels_disagree_in_series': np.mean([len(set(S['level_each'])) > 1 for S in res if S['status'] == 'ok'])
            if any(S['status'] == 'ok' for S in res) else np.nan,
        })
        print(f'{j}/{len(man)} {time.time() - t0:.0f} c', end='\r', flush=True)
    D = pd.DataFrame(rows)
    both = D[D.single_ok & D.series_ok]          # одни и те же пациенты для обоих сравнений
    summary = {
        'patients': len(D), 'k': a.k, 'seed': a.seed,
        'photos_passed_qc': round(float(D.photos_ok.mean()), 3),
        'series_answered': round(float(D.series_ok.mean()), 3),
        'paired_patients': len(both),
        'flip_single': round(float(both.single_flip.mean()), 3) if len(both) else None,
        'flip_series': round(float(both.series_flip.mean()), 3) if len(both) else None,
        'median_dP_single': round(float(both.single_dp.median()), 3) if len(both) else None,
        'median_dP_series': round(float(both.series_dp.median()), 3) if len(both) else None,
        'series_with_disagreeing_photos': (round(float(D.levels_disagree_in_series.mean()), 3)
                                           if D.levels_disagree_in_series.notna().any() else None),
        'caveat': 'модель обучена на этих пациентах и на таком же свете; симуляция, не реальные серии; точность не измерена',
    }
    out = os.path.join(a.data, 'results', 'series_stability')
    os.makedirs(out, exist_ok=True)
    D.to_csv(os.path.join(out, f'per_patient_k{a.k}_s{a.seed}.csv'), index=False)
    with open(os.path.join(out, f'summary_k{a.k}_s{a.seed}.json'), 'w') as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    print()
    print('| показатель | один снимок | серия из', a.k, '|')
    print('|---|---|---|')
    print(f"| смена уровня риска между двумя попытками | {summary['flip_single']} | {summary['flip_series']} |")
    print(f"| медиана |ΔP| между попытками | {summary['median_dP_single']} | {summary['median_dP_series']} |")
    print(f"пациентов {summary['patients']}, в паре оба ответа есть у {summary['paired_patients']}; "
          f"фото прошли QC: {summary['photos_passed_qc']:.0%}; серия дала ответ: {summary['series_answered']:.0%}; "
          f"серий, где фото дают разные уровни: {summary['series_with_disagreeing_photos']}")
    print('Ограничение:', summary['caveat'])
    print('Файлы:', out)


if __name__ == '__main__':
    main()
