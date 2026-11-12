"""Ошибки ядра по подгруппам: страна, пол, страна × пол — из OOF прогона evaluate.py --task core (ничего не обучает).

Зачем: в общих цифрах (AUC, чувств./спец.) не видно, у кого ложный риск. Здесь для каждой подгруппы — сколько
здоровых (Hb ≥ нормы ВОЗ) получили «средний» или «высокий» (FP), сколько больных получили «низкий» (FN),
чувствительность / специфичность зон (Уилсон 95%), AUC, MAE и сдвиг Hb (прогноз − анализ), доля отказов.
Отказ — снимок есть в манифесте, но ни одна модель его не оценила (маршрут: нет тканей / качество) или OOD.

    python scripts/subgroup_report.py --data <data>            # последние OOF <data>/results/oof_<tag>_*_core.csv
    python scripts/subgroup_report.py --data <data> --run <data>/results/runs/<run_id>
Страна — только у Eyes-defy (patient_id ED_<страна>_<номер>); OOF и манифест — вне git (медицинские данные).
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from anemia import metrics as M  # noqa: E402


def _country(man):
    c = man.patient_id.str.split('_').str[1]
    return c.where(man.dataset == 'eyes_defy', man.dataset)


def _row(name, G, n_all):
    """G — оценённые (принятые) снимки подгруппы; n_all — снимков подгруппы в манифесте."""
    an = G.anemic.values.astype(bool)
    lv = G.level.values.astype(int)
    err = G.hb_pred.values - G.hb_g_l.values
    h, a = ~an, an
    sens = M.wilson(int((a & (lv >= 1)).sum()), int(a.sum()))
    spec = M.wilson(int((h & (lv == 0)).sum()), int(h.sum()))
    nan = float('nan')
    return {'группа': name, 'n_всего': int(n_all), 'оценено': len(G), 'отказ': int(n_all - len(G)),
            'доля_отказов': (n_all - len(G)) / n_all if n_all else nan,
            'анемия': int(a.sum()), 'здоровых': int(h.sum()),
            'FP_средний_плюс': int((h & (lv >= 1)).sum()), 'FP_высокий': int((h & (lv == 2)).sum()),
            'FN_низкий': int((a & (lv == 0)).sum()),
            'чувств': sens[0], 'чувств_lo': sens[1], 'чувств_hi': sens[2],
            'спец': spec[0], 'спец_lo': spec[1], 'спец_hi': spec[2],
            'AUC': M.auc(G.p.values, an) if 0 < a.sum() < len(G) else nan,
            'MAE_Hb': float(np.abs(err).mean()) if len(G) else nan,
            'сдвиг_Hb': float(err.mean()) if len(G) else nan,
            'сдвиг_Hb_здоровых': float(err[h].mean()) if h.any() else nan,
            'сдвиг_Hb_анемия': float(err[a].mean()) if a.any() else nan}


def subgroup_table(oof, man):
    """oof — OOF всех моделей ядра одного датасета (group, anemic, level, p, hb_g_l, hb_pred, accepted);
    man — строки манифеста того же датасета (patient_id, group, sex, dataset). → таблица по подгруппам."""
    O = oof[oof.accepted.astype(bool)]
    assert not O.group.duplicated().any(), 'пациент оценён двумя моделями — OOF разных прогонов смешаны'
    m = man.drop_duplicates('group').assign(страна=lambda d: _country(d), пол=lambda d: d.sex.fillna('?'))
    O = O.merge(m[['group', 'страна', 'пол']], on='group', how='left')
    groups = [('все', m.group, O)]
    for col, title in (('страна', 'страна'), ('пол', 'пол')):
        for v in sorted(m[col].unique()):
            groups.append((f'{title}: {v}', m.group[m[col] == v], O[O[col] == v]))
    for (c, s), g in m.groupby(['страна', 'пол']):
        groups.append((f'страна × пол: {c} {s}', g.group, O[(O.страна == c) & (O.пол == s)]))
    return pd.DataFrame([_row(name, G, len(set(gr))) for name, gr, G in groups])


def latest_oof(data, run=None):
    pat = os.path.join(run, 'oof_*_core.csv') if run else os.path.join(data, 'results', 'oof_*_core.csv')
    files = sorted(glob.glob(pat))
    if not files:
        sys.exit(f'нет OOF ядра ({pat}) — сначала python scripts/evaluate.py --task core')
    return files


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('ANEMIA_DATA', 'data'))
    ap.add_argument('--run', help='папка прогона <data>/results/runs/<run_id> (по умолчанию — последние OOF)')
    ap.add_argument('--dataset', default='eyes_defy')
    ap.add_argument('--out', help='CSV таблицы (по умолчанию рядом с OOF, вне git)')
    a = ap.parse_args()
    files = [f for f in latest_oof(a.data, a.run) if f'_{a.dataset}' in os.path.basename(f)]
    oof = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    man = pd.read_csv(os.path.join(a.data, 'manifest.csv'))
    man = man[man.dataset == a.dataset]
    T = subgroup_table(oof, man)
    out = a.out or os.path.join(os.path.dirname(files[0]), f'subgroups_{a.dataset}.csv')
    T.to_csv(out, index=False)
    print('OOF: ' + ', '.join(os.path.basename(f) for f in files))
    print('FP — здоровый (Hb ≥ нормы ВОЗ) получил «средний» или «высокий»; FN — больной получил «низкий»; '
          'сдвиг Hb = прогноз − анализ, г/л. Подгруппы малы — смотреть ДИ.')
    pct = lambda x, lo, hi: '—' if x != x else f'{100 * x:.0f} [{100 * lo:.0f}–{100 * hi:.0f}]'
    num = lambda x: '—' if x != x else f'{x:.2f}'
    hb = lambda x: '—' if x != x else f'{x:+.1f}'
    print('| Группа | n / оценено / отказ | анемия / здоровых | FP средн.+ (из них высокий) | FN низкий | '
          'Чувств., % | Спец., % | AUC | MAE Hb | сдвиг Hb: все / здоровые / анемия |')
    print('|---|---|---|---|---|---|---|---|---|---|')
    for r in T.to_dict('records'):
        print(f"| {r['группа']} | {r['n_всего']} / {r['оценено']} / {r['отказ']} | {r['анемия']} / {r['здоровых']} | "
              f"{r['FP_средний_плюс']} ({r['FP_высокий']}) | {r['FN_низкий']} | "
              f"{pct(r['чувств'], r['чувств_lo'], r['чувств_hi'])} | {pct(r['спец'], r['спец_lo'], r['спец_hi'])} | "
              f"{num(r['AUC'])} | {'—' if r['MAE_Hb'] != r['MAE_Hb'] else format(r['MAE_Hb'], '.1f')} | "
              f"{hb(r['сдвиг_Hb'])} / {hb(r['сдвиг_Hb_здоровых'])} / {hb(r['сдвиг_Hb_анемия'])} |")
    print(f'→ {out}')


if __name__ == '__main__':
    main()
