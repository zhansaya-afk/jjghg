"""Обучает итоговую модель ядра HemaVision на ВСЕХ пациентах каждого датасета и сохраняет её в один файл.

    python scripts/train_core.py --data data                 # → models/hemavision_core.joblib (или $ANEMIA_MODELS)

Модель та же, что проверяет `python scripts/evaluate.py --task core` (anemia/core.py, fit_core): цифры точности
брать оттуда, этот скрипт метрик не считает. Пороги риска и решение «показывать ли Hb» подбираются вложенной CV
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
from evaluate import git_state, v12_features  # noqa: E402

from anemia import core as CORE  # noqa: E402
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
    a = ap.parse_args()
    man = pd.read_csv(os.path.join(a.data, 'manifest.csv'))
    # та же соль кэша, что у evaluate.py: признаки, посчитанные там, переиспользуются
    cache = FeatureCache(os.path.join(a.data, 'cache', 'features'), 'off' if a.no_cache else 'on',
                         salt={'evaluate_py': source_version(os.path.join(ROOT, 'scripts', 'evaluate.py'))})
    models = {}
    for ds in (CORE.SPECS if a.dataset == 'all' else [a.dataset]):
        df = man[man.dataset == ds].reset_index(drop=True)
        if not len(df):
            print(f'{ds}: нет в манифесте — пропущен')
            continue
        print(f'{ds}: признаки {len(df)} снимков...', flush=True)
        F, _ = v12_features(a.data, df, a.jobs, cache)
        F = CORE.usable(F)
        B = CORE.fit_core(F, CORE.feature_list(F.columns, ds), ds)
        models[ds] = B
        print(f"   пациентов {B['n_patients']} (анемия {B['n_anemic']}), без конъюнктивы {len(df) - B['n_patients']}; "
              f"пороги P(анемия): низкий ≤ {B['t_lo']:.2f}, высокий ≥ {B['t_hi']:.2f}; Hb: MAE вложенной CV "
              f"{B['hb_mae_cv']:.1f} г/л против константы {B['hb_mae_const_cv']:.1f} → "
              f"{'показывать ±' + format(B['hb_halfwidth'], '.0f') + ' г/л' if B['hb_ok'] else 'не показывать'}")
    if not models:
        sys.exit('нечего обучать')
    git = git_state()
    bundle = {'models': models, 'commit': git['commit'], 'created': dt.datetime.now().isoformat(timespec='seconds'),
              'sklearn': sklearn.__version__, 'manifest_rows': len(man)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    joblib.dump(bundle, a.out)
    print(f'Модель → {a.out} (коммит {git["commit"]}, sklearn {sklearn.__version__})')


if __name__ == '__main__':
    main()
