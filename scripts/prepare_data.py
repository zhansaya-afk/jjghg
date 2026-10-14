"""Собирает data/manifest.csv из всех найденных датасетов и проверяет отсутствие утечки между фолдами.

    python scripts/prepare_data.py [--data data] [--folds 5] [--seed 0]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anemia.data import build_manifest, check_no_leak  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('ANEMIA_DATA', 'data'))
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    df = build_manifest(a.data, n_folds=a.folds, seed=a.seed)
    cross = check_no_leak(df)
    out = os.path.join(a.data, 'manifest.csv')
    df.to_csv(out, index=False)
    print(f'Сохранено: {out}')
    s = df.groupby('dataset').agg(снимков=('image', 'size'), групп=('group', 'nunique'),
                                  с_маской=('mask_conj', lambda m: int(m.notna().sum())),
                                  анемия=('anemic', 'sum'),
                                  Hb_мин=('hb_g_l', 'min'), Hb_медиана=('hb_g_l', 'median'), Hb_макс=('hb_g_l', 'max'))
    print(s.to_string())
    print('Фолды (снимков):', df.groupby(['dataset', 'fold']).size().unstack(fill_value=0).to_dict('index'))
    dups = len(df) - df.group.nunique() - (len(df) - df.patient_id.nunique())
    print(f'Почти-дублей склеено в группы: {max(dups, 0)}; групп сразу в нескольких датасетах: {len(cross)}')


if __name__ == '__main__':
    main()
