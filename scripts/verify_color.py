"""Сверка формул цвета с эталоном (CIE / IEC / Гонсалес) и график стабильности признаков при другом свете.

    python scripts/verify_color.py --out results/color_check

Пишет verify.csv (таблица сверки), stability.csv, spread.csv и stability.png. Код возврата 1 — сверка не прошла.
Свет моделируется (anemia/augment.py): это проверка формул, не точность на реальных снимках.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd  # noqa: E402

from anemia import color_check as K  # noqa: E402

BLUE, ORANGE, INK, MUTED, GRID = '#2a78d6', '#eb6834', '#0b0b0b', '#52514e', '#e4e3df'


def plot(rows, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    T = pd.DataFrame(rows)
    temp = T[T['экспозиция'] == 1.0]
    expo = T[T['T'] == 6500]
    feats = [('EI', 'Erythema Index'), ('a*', 'a* (CIELAB)'), ('оттенок', 'оттенок HSI, °')]
    fig, ax = plt.subplots(2, 3, figsize=(13, 7.2), dpi=150)
    for j, (f, title) in enumerate(feats):
        for i, (g, xcol, xlab) in enumerate(((temp, 'T', 'цветовая температура, K'),
                                              (expo, 'экспозиция', 'экспозиция, ×'))):
            a = ax[i, j]
            for how, col in (('сырой', ORANGE), ('баланс по склере', BLUE)):
                for tis, ls in (('конъюнктива, красная', '-'), ('конъюнктива, бледная', '--')):
                    d = g[(g['способ'] == how) & (g['ткань'] == tis)].sort_values(xcol)
                    a.plot(d[xcol], d[f], ls, color=col, lw=2, label=f'{how}, {tis.split(", ")[1]}')
            a.set_title(title, color=INK, fontsize=11, loc='left')
            a.set_xlabel(xlab, color=MUTED, fontsize=9)
            a.grid(color=GRID, lw=0.8)
            for s in ('top', 'right'):
                a.spines[s].set_visible(False)
            a.tick_params(colors=MUTED, labelsize=8)
            if xcol == 'T':
                a.axvspan(2700, 4000, color=GRID, alpha=0.5, lw=0)
    ax[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle('Признаки цвета при смоделированном свете: сплошная — красная конъюнктива, пунктир — бледная. '
                 'Если линии одного цвета пересекаются, свет сильнее сигнала бледности', fontsize=10, color=INK)
    fig.text(0.01, 0.005, 'Серая зона 2700–4000 K: красный канал упирается в 1 (клиппинг), баланс по склере не '
             'восстанавливает цвет. Модель света — фон Крис, баланс белого камеры на D65 (худший случай).',
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(path)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', default='results/color_check')
    a = ap.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    V = pd.DataFrame(K.verify())
    V.to_csv(os.path.join(a.out, 'verify.csv'), index=False)
    S = K.stability()
    pd.DataFrame(S).to_csv(os.path.join(a.out, 'stability.csv'), index=False)
    sp = pd.DataFrame(K.spread(S))
    sp.to_csv(os.path.join(a.out, 'spread.csv'), index=False)
    plot(S, os.path.join(a.out, 'stability.png'))
    with pd.option_context('display.width', 200, 'display.max_colwidth', 80):
        print(V[['проверка', 'макс_ошибка', 'допуск', 'прошла']].to_string(index=False))
        print(sp.round(3).to_string(index=False))
    ok = bool(V['прошла'].all())
    print('сверка:', 'пройдена' if ok else 'НЕ пройдена', f'({int(V["прошла"].sum())}/{len(V)})')
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
