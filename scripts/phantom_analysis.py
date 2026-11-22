"""Фантом-эксперимент: печатная карточка + шкала образцов → a*, EI до/после коррекции по карточке.

Проверяет камеру и коррекцию цвета (порядок образцов, разброс при другом свете/телефоне). НЕ проверяет связь с Hb,
сегментацию и модель риска; модель не трогает. Подробно: anemia/phantom.py, docs/phantom_experiment.md (протокол),
docs/phantom/README.md (результат 2026-11-22).

    python scripts/phantom_analysis.py card --out phantom_card.pdf            # лист A4 для печати (+ PNG рядом)
    python scripts/phantom_analysis.py analyze --photos DIR --out DIR_OUT      # DIR/p1/okno/*.jpg или p1_okno_01.jpg
    python scripts/phantom_analysis.py analyze --photos DIR --out DIR_OUT --gelatin   # + круги G1–G8 с желатином
    python scripts/phantom_analysis.py analyze --photos DIR --out DIR_OUT --exploratory  # разведочно (после данных)
    python scripts/phantom_analysis.py demo --out DIR_OUT                      # синтетика: только проверка кода

Выход analyze: samples.csv (каждый образец × снимок × коррекция), summary.csv, report.md, phantom_plot.png,
skipped.csv (снимки, где не нашёлся лист, и почему).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from anemia import phantom as P  # noqa: E402

IMG_EXT = ('.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp')
RAW_EXT = ('.heic', '.heif', '.dng')
COLS = ['method', 'metric', 'n_photos', 'monotone', 'rho_min', 'sd_step', 'light_sd_step', 'phone_sd_step',
        'cv_pct', 'card_dE', 'n_clipped_samples', 'n_out_of_gamut']
METHOD_RU = {'raw': 'как снято', 'wb': 'баланс белого по серым', 'ccm': 'матрица по карточке'}


def load_rgb(path):
    from PIL import Image, ImageOps
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert('RGB'))


def list_photos(photos):
    """Снимки в папке: photos/<телефон>/<свет>/*.jpg или плоско photos/p1_okno_01.jpg. → [(путь, phone, light, rep)]."""
    out = []
    for dirpath, _, files in os.walk(photos):
        rel = os.path.relpath(dirpath, photos).replace('\\', '/')
        parts = [] if rel == '.' else rel.split('/')
        for i, f in enumerate(sorted(files)):
            if not f.lower().endswith(IMG_EXT + RAW_EXT):
                continue
            if len(parts) >= 2:
                phone, light, rep = parts[-2].lower(), parts[-1].lower(), i + 1
            else:
                phone, light, rep = P.parse_name(f)
            out.append((os.path.join(dirpath, f), phone, light, rep))
    return sorted(out)


def analyze_dir(photos, kinds, min_markers=3):
    rows, skipped = [], []
    for path, phone, light, rep in list_photos(photos):
        f = os.path.relpath(path, photos)
        if path.lower().endswith(RAW_EXT):
            skipped.append({'photo': f, 'reason': 'HEIC/DNG не читается: снимайте в JPEG (см. инструкцию)'})
            continue
        try:
            d = P.analyze_image(load_rgb(path), kinds=kinds, min_markers=min_markers)
        except Exception as e:                                          # noqa: BLE001 — причина в skipped.csv
            skipped.append({'photo': f, 'reason': str(e)})
            print(f'  пропущен {f}: {e}')
            continue
        if phone == '?':
            print(f'  {f}: телефон/свет не понятны из имени — положите в папку <телефон>/<свет>/')
        d.insert(0, 'photo', f)
        d.insert(1, 'phone', phone)
        d.insert(2, 'light', light)
        d.insert(3, 'rep', rep)
        rows.append(d)
        if 'error' in d and d.error.notna().any():
            bad = d[d.error.notna()]
            print(f'  {f}: {"; ".join(sorted(set(bad.method + ": " + bad.error)))}')
    return (pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()), pd.DataFrame(skipped)


def plot(df, kind, path):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('  matplotlib нет — график пропущен')
        return None
    d = df[(df.kind == kind) & df.name.notna()].copy()
    d['idx'] = d.name.map(P._order)
    lights = sorted(d.light.unique())
    phones = sorted(d.phone.unique())
    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for j, m in enumerate(P.METHODS):
        for i, met in enumerate(('a', 'EI')):
            a = ax[i, j]
            for ph, g in d[d.method == m].groupby('photo'):
                g = g.sort_values('idx')
                a.plot(g.idx, g[met], color=cmap(lights.index(g.light.iloc[0]) % 10),
                       ls=['-', '--', ':', '-.'][phones.index(g.phone.iloc[0]) % 4], lw=1, alpha=0.8)
            a.set_title(f'{met if met == "EI" else "a*"} — {METHOD_RU[m]}', fontsize=10)
            a.grid(alpha=0.3)
            if i == 1:
                a.set_xlabel('образец (1 — бледный … 8 — тёмно-красный)')
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=cmap(k % 10), label=f'свет: {lt}') for k, lt in enumerate(lights)]
    h += [Line2D([], [], color='k', ls=['-', '--', ':', '-.'][k % 4], label=f'телефон: {p}')
          for k, p in enumerate(phones)]
    fig.legend(handles=h, loc='lower center', ncol=min(len(h), 6), fontsize=9)
    fig.suptitle('Фантом: каждая линия — один снимок. Хорошо = линии растут и лежат друг на друге', fontsize=11)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def report_md(summary, kind, n_files, skipped, synthetic, exploratory=False):
    s = summary[COLS].copy()
    s['method'] = s.method.map(lambda m: f'{m} ({METHOD_RU[m]})')
    head = ['# Фантом-эксперимент: итог', '']
    if exploratory:
        head += [f'**РАЗВЕДОЧНЫЙ АНАЛИЗ — режим выбран после просмотра данных (лист по ≥ '
                 f'{P.EXPLORATORY["min_markers"]} маркерам, без {", ".join(P.EXPLORATORY["drop"])}). Это гипотеза, '
                 'не подтверждение; вердикт ниже справочный и не заменяет основной анализ.**', '']
    if synthetic:
        head += ['**СИНТЕТИКА — только проверка, что код работает. Это не результат эксперимента.**', '']
    head += [f'Образцы: {"печатная шкала S1–S8" if kind == "sample" else "желатин G1–G8"}. '
             f'Снимков в папке: {n_files}, пропущено: {len(skipped)}.', '',
             f'**Вердикт (критерии заданы до съёмки: порядок верен ≥ {P.CRITERIA["monotone_min"]:.0%} снимков, '
             f'разброс ≤ {P.CRITERIA["sd_step_max"]} шага, карточка применена ≥ {P.CRITERIA["coverage_min"]:.0%} '
             f'снимков, карточка уменьшает разброс):** {P.verdict(summary, n_files)}', '',
             f'Версия: {P.VERSION}, код {code_version()}. Критерии и параметры (обрезка {P.CLIP_HI}/{P.CLIP_LO}, '
             f'центральные 60% поля, ≥ {P.MIN_FIT_PATCHES} полей для матрицы, образец с > {P.MAX_CLIPPED:.0%} '
             f'пересвета — не в сводку) зафиксированы до съёмки.', '',
             '| ' + ' | '.join(s.columns) + ' |', '|' + '---|' * len(s.columns)]
    for _, r in s.iterrows():
        head.append('| ' + ' | '.join(f'{v:.2f}' if isinstance(v, float) else str(v) for v in r) + ' |')
    head += ['', 'Как читать:',
             '- `monotone` — доля снимков, где a*/EI растут строго от S1 к S8; `rho_min` — худший Спирмен.',
             '- `sd_step` — разброс одного образца по всем снимкам в долях шага между соседними образцами. '
             '< 0,5 — соседние образцы различимы на любом снимке; > 1 — свет/телефон сдвигает цвет сильнее, '
             'чем отличаются соседние образцы.',
             '- `light_sd_step` — разброс из-за света (внутри телефона), `phone_sd_step` — из-за телефона (при одном '
             'свете). `cv_pct` — SD/среднее, %. `card_dE` — остаточная ΔE карточки после коррекции (на тех же '
             'полях, оптимистично; у raw — насколько снимок далёк от номинала).',
             f'- Шаг шкалы: a* ≈ {summary[(summary.method == "ccm") & (summary.metric == "a")].step.mean():.1f}, '
             'EI ≈ ' + f'{summary[(summary.method == "ccm") & (summary.metric == "EI")].step.mean():.3f} на образец. '
             '«Различимы» относится только к такому шагу; разница «норма/анемия» у конъюнктивы может быть меньше. '
             'Это не чувствительность к Hb.',
             '- Ограничение: фантом проверяет камеру и коррекцию цвета, **не связь цвета с Hb** и не модель риска. '
             'Бумага не просвечивает, как ткань; принтер задаёт свои абсолютные цвета.']
    return '\n'.join(head) + '\n'


def code_version():
    import subprocess
    try:
        r = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True,
                           timeout=10)
        dirty = subprocess.run(['git', '-C', ROOT, 'status', '--porcelain', 'anemia/phantom.py',
                                'scripts/phantom_analysis.py'], capture_output=True, text=True, timeout=10).stdout
        return r.stdout.strip() + ('-dirty' if dirty.strip() else '') if r.returncode == 0 else '?'
    except (OSError, subprocess.SubprocessError):
        return '?'


def run_analysis(df, skipped, n_files, out, kind, synthetic=False, exploratory=False):
    os.makedirs(out, exist_ok=True)
    df = df.assign(synthetic=synthetic, exploratory=exploratory)
    df.to_csv(os.path.join(out, 'samples.csv'), index=False)
    skipped.to_csv(os.path.join(out, 'skipped.csv'), index=False)
    d = df[df.kind == kind]
    if exploratory:
        d = d[~d.name.isin(P.EXPLORATORY['drop'])]
    summary = P.summarize(d)
    summary.assign(synthetic=synthetic, exploratory=exploratory).to_csv(os.path.join(out, 'summary.csv'),
                                                                        index=False)
    md = report_md(summary, kind, n_files, skipped, synthetic, exploratory)
    with open(os.path.join(out, 'report.md'), 'w', encoding='utf-8') as fh:
        fh.write(md)
    plot(df, kind, os.path.join(out, 'phantom_plot.png'))
    print(md)
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('card')
    c.add_argument('--out', default='phantom_card.pdf')
    c.add_argument('--dpi', type=int, default=300)
    a = sub.add_parser('analyze')
    a.add_argument('--photos', required=True)
    a.add_argument('--out', required=True)
    a.add_argument('--gelatin', action='store_true', help='анализировать круги G1–G8 (желатин), а не S1–S8')
    a.add_argument('--exploratory', action='store_true',
                   help='разведочный режим (выбран после просмотра данных 22.11): лист по 2 маркерам, без S8')
    dm = sub.add_parser('demo')
    dm.add_argument('--out', required=True)
    args = ap.parse_args(argv)

    if args.cmd == 'card':
        P.save_sheet_pdf(args.out, args.dpi)
        from PIL import Image
        Image.fromarray(P.render_sheet(args.dpi)).save(os.path.splitext(args.out)[0] + '.png')
        print(f'готово: {args.out} — печатать A4, масштаб 100% («реальный размер»), проверить линейку 50 мм')
        return 0
    if args.cmd == 'analyze':
        kind = 'gel' if args.gelatin else 'sample'
        mm = P.EXPLORATORY['min_markers'] if args.exploratory else 3
        df, skipped = analyze_dir(args.photos, (kind,), mm)
        n = len(list_photos(args.photos))
        if df.empty:
            print('ни один снимок не разобран — см. skipped.csv')
            os.makedirs(args.out, exist_ok=True)
            skipped.to_csv(os.path.join(args.out, 'skipped.csv'), index=False)
            return 1
        run_analysis(df, skipped, n, args.out, kind, exploratory=args.exploratory)
        return 0
    # demo: 2 «телефона» × 3 света × 2 повтора на синтетике
    sheet = P.render_sheet(150)
    rows = []
    for ph in ('p1', 'p2'):
        for light, (T, e) in {'okno': (6000, 1.0), 'lampa': (3200, 0.9), 'led': (5000, 1.1)}.items():
            for rep in (1, 2):
                d = P.analyze_image(P.synthetic_photo(sheet, T, e, ph, seed=rep * 100 + T))
                d.insert(0, 'photo', f'{ph}_{light}_{rep:02d}.jpg')
                d.insert(1, 'phone', ph)
                d.insert(2, 'light', light)
                d.insert(3, 'rep', rep)
                rows.append(d)
    run_analysis(pd.concat(rows, ignore_index=True), pd.DataFrame(), 12, args.out, 'sample', synthetic=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
