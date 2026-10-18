"""Готовит итоговую модель ядра HemaVision (все пациенты каждого датасета) в одном файле — обучает только то,
чего нет: 1) совместимая модель уже лежит в --out (тот же код признаков и scikit-learn) → ничего не делаем;
2) модели взрослых есть в GitHub Release для текущего кода (anemia/model_store.py) → скачиваем, сверяем sha256;
3) остальное (дети CP-AnemiC — всегда, лицензия не даёт публиковать) обучаем здесь. --force — обучить всё заново.

    python scripts/train_core.py --data data                 # → models/hemavision_core.joblib (или $ANEMIA_MODELS)

Модель та же, что проверяет `python scripts/evaluate.py --task core` (anemia/core.py, fit_core): цифры точности
брать оттуда, этот скрипт метрик не считает. Конъюнктиву снимков глаза выделяет сегментатор по маскам врачей
(anemia/segment_baseline.py): признаки обучения — по маскам моделей, не видевших фолда снимка (как в evaluate),
в файл модели (conj_seg) идёт сегментатор на всех снимках — им predict размечает новые фото.
Пороги риска и решение «показывать ли Hb» подбираются вложенной CV
по пациентам внутри обучающих данных. Признаки — общий кэш с evaluate.py (<data>/cache/features).
"""
import argparse
import datetime as dt
import os
import sys

import joblib
import pandas as pd
import sklearn

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluate import _frame, conj_segmenters, core_jobs, git_state, route_counts, v12_features  # noqa: E402

from anemia import core as CORE  # noqa: E402
from anemia import model_store as STORE  # noqa: E402
from anemia.cache import FeatureCache, source_version  # noqa: E402
from anemia.predict import MODEL_FILE  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('ANEMIA_DATA', 'data'))
    ap.add_argument('--out', default=os.path.join(os.environ.get('ANEMIA_MODELS', os.path.join(ROOT, 'models')),
                                                  MODEL_FILE))
    ap.add_argument('--dataset', choices=['all', *CORE.SPECS], default='all')
    ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--no-cache', action='store_true')
    ap.add_argument('--force', action='store_true', help='обучить всё заново, не глядя на готовые модели')
    ap.add_argument('--no-download', action='store_true', help='не скачивать опубликованную модель взрослых')
    ap.add_argument('--publish-dir', help='сюда же положить файл для GitHub Release (только модели взрослых)')
    a = ap.parse_args()
    man = pd.read_csv(os.path.join(a.data, 'manifest.csv'))
    # та же соль кэша, что у evaluate.py: признаки, посчитанные там, переиспользуются
    cache = FeatureCache(os.path.join(a.data, 'cache', 'features'), 'off' if a.no_cache else 'on',
                         salt={'evaluate_py': source_version(os.path.join(ROOT, 'scripts', 'evaluate.py'))})
    keys = list(CORE.SPECS) if a.dataset == 'all' else [a.dataset]
    models, feats, sources, conj_seg = {}, {}, {}, None
    if not a.force and os.path.exists(a.out):
        old = joblib.load(a.out)
        why = STORE.check_compat(old.get('compat'))
        if why:
            print(f'Модель {a.out} не подходит ({why}) — готовлю заново')
        else:
            conj_seg = old.get('conj_seg') if not ({'eyes_defy', 'eyes_defy_conj', 'eyes_defy_sclera'} & set(keys)
                                                    - set(old['models'])) else None
            models.update({k: v for k, v in old['models'].items()
                           if k in keys and (conj_seg is not None or CORE.SPECS[k]['data'] == 'cp_anemic')})
            sources.update(old.get('sources', {}))
    adult = [k for k in keys if k not in models and CORE.SPECS[k]['data'] in STORE.PUBLIC_DATASETS]
    if adult and not a.force and not a.no_download:
        path, msg = STORE.fetch_published(os.path.dirname(os.path.abspath(a.out)))
        print(msg)
        if path:
            pub = joblib.load(path)
            got = {k: v for k, v in pub['models'].items() if k in adult} if pub.get('conj_seg') is not None else {}
            conj_seg = pub.get('conj_seg') if got else conj_seg
            models.update(got)
            sources.update({k: f"GitHub Release models-{pub['compat']['code']} (коммит {pub.get('commit')})" for k in got})
    todo = [k for k in keys if k not in models]
    if not todo and not a.publish_dir:
        print(f'Модель готова, обучение не нужно: {a.out} ({", ".join(sorted(models))})')
        return
    for key in todo:
        ds = CORE.SPECS[key]['data']
        df = man[man.dataset == ds].reset_index(drop=True)
        if not len(df):
            print(f'{key}: нет {ds} в манифесте — пропущен')
            continue
        if ds not in feats:
            if ds == 'cp_anemic':
                print(f'{ds}: признаки {len(df)} снимков...', flush=True)
                feats[ds] = v12_features(a.data, df, a.jobs, cache)[0]
            else:                        # снимки глаза: обучаемся только на тех, что примет анализ фото
                if not df.mask_conj.notna().all():
                    sys.exit(f'{ds}: у части снимков нет масок врачей — сегментатор конъюнктивы обучить не на чем')
                print(f'{ds}: сегментатор конъюнктивы по маскам врачей {len(df)} снимков...', flush=True)
                segs = conj_segmenters(a.data, df, a.jobs, cache)
                conj_seg = segs[()]
                J = core_jobs(a.data, df, a.jobs, segs, cache)
                print(f'{ds}: маршрут анализа фото: ' + ', '.join(f'{k} {v}' for k, v in route_counts(J, len(df)).items()))
                feats[ds] = _frame(J, 'test')
        F = CORE.usable(feats[ds], key)
        B = CORE.fit_core(F, CORE.feature_list(F.columns, key), key)
        models[key] = B
        sources[key] = 'обучена здесь'
        print(f"{key}: пациентов {B['n_patients']} (анемия {B['n_anemic']}), без нужной ткани {len(df) - B['n_patients']}; "
              f"пороги P(анемия): низкий ≤ {B['t_lo']:.2f}, высокий ≥ {B['t_hi']:.2f}; Hb: MAE вложенной CV "
              f"{B['hb_mae_cv']:.1f} г/л против константы {B['hb_mae_const_cv']:.1f} → "
              f"{'показывать ±' + format(B['hb_halfwidth'], '.0f') + ' г/л' if B['hb_ok'] else 'не показывать'}")
    if not models:
        sys.exit('нечего обучать')
    git = git_state()
    bundle = {'models': models, 'commit': git['commit'], 'created': dt.datetime.now().isoformat(timespec='seconds'),
              'sklearn': sklearn.__version__, 'manifest_rows': len(man), 'compat': STORE.compat_stamp(),
              'sources': sources, 'conj_seg': conj_seg}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    joblib.dump(bundle, a.out)
    print(f'Модель → {a.out} (коммит {git["commit"]}, sklearn {sklearn.__version__}); '
          + '; '.join(f'{k}: {v}' for k, v in sorted(sources.items())))
    if a.publish_dir:
        path, tag = STORE.write_published(bundle, a.publish_dir)
        print(f'Для публикации: {path} (+ .json) → GitHub Release с тегом {tag}')


if __name__ == '__main__':
    main()
