"""Где теряется точность v12 на Eyes-defy: маска врача (потолок) против автоматических масок на тех же фолдах.

    python scripts/error_analysis.py --data data --tag ea_v12

Источники маски конъюнктивы (склера везде своя — anemia.regions, как в демо):
  doctor — разметка врача Eyes-defy (верхняя граница: идеальная сегментация);
  auto   — anemia.regions (v12 демо, без обучения);
  v11    — попиксельный HGB v11, обученный на масках врачей только обучающих фолдов (OOF).
Модель, признаки, обучающий свет, порог P(анемия) — ровно как в `evaluate.py --task v12`.
Сравнение — на одних пациентах (у кого ткань нашлась всеми источниками), разница с бутстрэп-ДИ по пациентам.

Выход: results/errors/<tag>/ — report.md, *.csv, worst/*.png. Это медицинские данные: папка в .gitignore,
не коммитить и не отправлять во внешние сервисы. Пороги по этим результатам не подбирать.
"""
import argparse
import datetime as dt
import os
import sys

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluate import run_seg, v12_cv  # noqa: E402

from anemia import errors as E  # noqa: E402
from anemia.cache import dataset_fingerprint  # noqa: E402
from anemia import metrics as M  # noqa: E402
from anemia import regions as R  # noqa: E402
from anemia import v12_model as V12  # noqa: E402
from anemia.augment import STRESS_CONDITIONS, apply_illuminant  # noqa: E402
from anemia.io import load_mask, load_rgb, resize_to  # noqa: E402
from anemia.labs import is_pregnant  # noqa: E402

SOURCES = ('doctor', 'auto', 'v11')
NAMES = {'doctor': 'маска врача', 'auto': 'авто v12 (regions)', 'v11': 'авто v11 (обучена на масках врачей)'}


def _job(root, row, v11_mask, seed):
    path = os.path.join(root, row['image'])
    src = load_rgb(path)
    rgb = load_rgb(path, V12.DETECT_SIDE)
    cv2.setRNGSeed(0)                                    # как в evaluate.py: GrabCut детерминирован
    det = R.detect_regions(rgb, 'both')                  # без MediaPipe: вырезка = весь кадр
    crop, auto, sclera = det['crop'], det['conj'], det['sclera']
    shape = crop.shape[:2]
    doc = resize_to(load_mask(os.path.join(root, row['mask_conj'])), shape, nearest=True)
    masks = {'doctor': doc, 'auto': auto, 'v11': resize_to(v11_mask, shape, nearest=True)}
    age = row['age'] if row['age'] == row['age'] else 30.0
    sex = row['sex'] if isinstance(row['sex'], str) else None
    y = V12.hb_class(row['hb_g_l'] / 10, *V12.who_thresholds(sex, age, is_pregnant(row['pregnant'])))
    base = {'group': row['group'], 'fold': int(row['fold']), 'anemic': bool(row['anemic']), 'y': y,
            'male': 1.0 if sex == 'M' else 0.0 if sex == 'F' else 0.5}
    rows, srows = [], []
    for s, conj in masks.items():
        scl = sclera & ~conj
        meta = {**base, 'source': s}
        rows.append({**meta, 'aug': 0, **V12.eye_features(crop, conj, scl)})
        rng = np.random.default_rng(seed)                # одни и те же обучающие копии света для всех масок
        for a in range(1, V12.N_RELIGHT + 1):
            rows.append({**meta, 'aug': a, **V12.eye_features(V12.relight(crop, rng), conj, scl)})
        srows += [{**meta, 'cond': f'{T}K×{e}', **V12.eye_features(apply_illuminant(crop, T, e), conj, scl)}
                  for T, e in STRESS_CONDITIONS]
    lab = cv2.cvtColor(np.ascontiguousarray(crop, np.float32), cv2.COLOR_RGB2Lab)
    roi = auto | sclera
    info = {'group': row['group'], 'country': row['patient_id'].split('_')[1], 'sex': sex or '?',
            'age': row['age'], 'hb_g_l': row['hb_g_l'], 'y': y,
            'doc_diff': E.mask_photo_diff(os.path.join(root, row['mask_conj']), src),
            'iou_auto': M.iou(auto, doc), 'iou_v11': M.iou(masks['v11'], doc),
            'conj_frac_doc': float(doc.mean()), 'conj_frac_auto': float(auto.mean()),
            'frame_L': float(np.median(lab[..., 0])),
            'sclera_L': float(np.median(lab[..., 0][sclera])) if sclera.any() else np.nan,
            'sclera_found': bool(sclera.any()),
            'blur': R.blur_ratio(crop, roi) if roi.any() else np.nan,
            'issues': '; '.join(R.quality(crop, auto, sclera, max(src.shape[:2])))}
    pics = {'crop': (crop * 255).astype(np.uint8), **masks, 'sclera': sclera}
    return rows, srows, info, pics


def ea_rows(man):
    return man[(man.dataset == 'eyes_defy') & man.mask_conj.notna()].reset_index(drop=True)


def ea_fingerprint(data, df):
    """Ключ кэша признаков: строки манифеста, содержимое снимков и масок, код пакета и обоих скриптов."""
    return dataset_fingerprint(data, df, ['image', 'mask_conj'],
                               extra=(os.path.abspath(__file__), os.path.join(ROOT, 'scripts', 'evaluate.py')))


def ea_cache_path(data, tag):
    return os.path.join(data, 'results', f'ea_cache_{tag}.pkl')


def ea_cache_fresh(data, tag, df):
    """Кэш есть и посчитан ровно для этих данных и этого кода (ключ в соседнем файле .key)."""
    cache = ea_cache_path(data, tag)
    if not (os.path.exists(cache) and os.path.exists(cache + '.key')):
        return False
    with open(cache + '.key') as f:
        return f.read().strip() == ea_fingerprint(data, df)


def fmt_pct(v):
    return '—' if v != v else f'{v * 100:.0f}'


def md_table(df, floatfmt='{:.3f}'):
    cols = list(df.columns)
    out = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for _, r in df.iterrows():
        out.append('| ' + ' | '.join(floatfmt.format(v) if isinstance(v, float) else str(v) for v in r) + ' |')
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('ANEMIA_DATA', 'data'))
    ap.add_argument('--tag', default='ea_v12')
    ap.add_argument('--top', type=int, default=20)
    ap.add_argument('--target-sens', type=float, default=0.80)
    ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--no-cache', action='store_true', help='пересчитать признаки, даже если есть кэш')
    a = ap.parse_args()
    out = os.path.join(ROOT, 'results', 'errors', a.tag)
    os.makedirs(os.path.join(out, 'worst'), exist_ok=True)
    man = pd.read_csv(os.path.join(a.data, 'manifest.csv'))
    df = ea_rows(man)
    cache = ea_cache_path(a.data, a.tag)                 # признаки + маски: повтор за минуты
    if not a.no_cache and ea_cache_fresh(a.data, a.tag, df):
        print(f'Признаки из кэша {cache} (пересчитать: --no-cache)', flush=True)
        res = pd.read_pickle(cache)
    else:
        if os.path.exists(cache):                        # данные, код или --no-cache: старый кэш не удаляем
            stale = cache[:-4] + f'.stale-{dt.datetime.now():%Y%m%d-%H%M%S}.pkl'
            os.replace(cache, stale)
            print(f'Старый кэш {cache} сохранён как {stale}, пересчитываю',
                  flush=True)
        print(f'Eyes-defy: {len(df)} пациентов, {df.fold.nunique()} фолдов. Обучаю v11-сегментацию (OOF)...',
              flush=True)
        seg, v11_masks, _ = run_seg(a.data, df, a.jobs)
        print(f'  v11 IoU {seg["iou_mean"]:.3f}. Ищу ткани v12 и считаю признаки для 3 масок...', flush=True)
        res = Parallel(n_jobs=a.jobs)(delayed(_job)(a.data, r, v11_masks[i], i)
                                      for i, r in enumerate(df.to_dict('records')))
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        pd.to_pickle(res, cache + '.tmp')
        os.replace(cache + '.tmp', cache)
        with open(cache + '.key', 'w') as f:
            f.write(ea_fingerprint(a.data, df))
    F = pd.DataFrame([x for rows, _, _, _ in res for x in rows])
    S = pd.DataFrame([x for _, srows, _, _ in res for x in srows])
    info = pd.DataFrame([x[2] for x in res])
    pics = {x[2]['group']: x[3] for x in res}
    sets = V12.feature_sets(F.columns, df.sex.notna().any())
    bad = set(info.group[info.doc_diff > 25])           # разметка врача не совпадает с фото — исключить везде
    if bad:
        print(f'  Испорченная маска врача, пациент исключён из сравнения: {sorted(bad)}', flush=True)
        F, S, info = F[~F.group.isin(bad)], S[~S.group.isin(bad)], info[~info.group.isin(bad)]

    # ------------------------------------------------ 1. маска врача vs автоматические
    oof, table = {}, []
    for mode in ('conj', 'both'):
        feats = sets[mode]
        ok = {s: set(F[(F.source == s) & (F.aug == 0)].dropna(subset=feats).group) for s in SOURCES}
        common = set.intersection(*ok.values())
        runs = [(s, ok[s], 'все с тканью') for s in SOURCES] + [(s, common, 'общие') for s in SOURCES]
        for s, groups, scope in runs:
            Fm = F[(F.source == s) & F.group.isin(groups)].dropna(subset=feats)
            Sm = S[(S.source == s) & S.group.isin(groups)].dropna(subset=feats)
            print(f'  v12 · {mode} · {s} · {scope}: {len(groups)} пациентов', flush=True)
            O, _ = v12_cv(Fm, Sm, feats, a.target_sens)
            O = O.merge(info[['group', 'hb_g_l']], on='group')
            oof[(mode, s, scope)] = O
            for rule, use_thr in (('argmax', False), ('порог 0.80', True)):
                r = E.summary(O, use_thr)
                k = E.decide(O, use_thr)
                low = O.hb_g_l.values < 100
                table.append({'режим': mode, 'маска': NAMES[s], 'пациенты': scope, 'правило': rule, 'n': r['n'],
                              'верный класс (сбаланс.), %': fmt_pct(r['bal_acc']), 'AUC': f'{r["auc"]:.3f}',
                              'чувств., %': fmt_pct(r['sens']), 'спец., %': fmt_pct(r['spec']),
                              'чувств. Hb<100, %': f'{fmt_pct((k[low] > 0).mean())} (n={low.sum()})'})
    T = pd.DataFrame(table)
    T.to_csv(os.path.join(out, 'masks_compare.csv'), index=False)
    deltas = []
    for mode in ('conj', 'both'):
        for s in ('auto', 'v11'):
            for rule, use_thr in (('argmax', False), ('порог 0.80', True)):
                d = E.paired_delta(oof[(mode, 'doctor', 'общие')], oof[(mode, s, 'общие')], use_thr)
                deltas.append({'режим': mode, 'врач минус': NAMES[s], 'правило': rule,
                               'Δ сбаланс. точность, п.п.': f'{d["d_bal_acc"] * 100:+.1f} '
                               f'[{d["d_bal_acc_ci"][0] * 100:+.1f}; {d["d_bal_acc_ci"][1] * 100:+.1f}]',
                               'Δ AUC': f'{d["d_auc"]:+.3f} [{d["d_auc_ci"][0]:+.3f}; {d["d_auc_ci"][1]:+.3f}]'})
    D = pd.DataFrame(deltas)

    # ------------------------------------------------ 2. матрицы ошибок (conj, общие пациенты)
    conf = []
    for s in ('doctor', 'auto'):
        O = oof[('conj', s, 'общие')]
        for rule, use_thr in (('argmax', False), ('порог 0.80', True)):
            C = E.confusion(O.y.values, E.decide(O, use_thr))
            C.to_csv(os.path.join(out, f'confusion_{s}_{"thr" if use_thr else "argmax"}.csv'))
            conf.append((f'{NAMES[s]} · {rule}', C))

    # ------------------------------------------------ 3. худшие случаи (авто-маска v12, conj, argmax)
    Oa, Od = oof[('conj', 'auto', 'общие')], oof[('conj', 'doctor', 'общие')].set_index('group')
    Oa = Oa.assign(err=E.expected_error(Oa), k=E.decide(Oa, False))
    Oa['p_anem'] = 1 - np.stack(Oa.P.values)[:, 0]
    Oa['k_doc'] = E.decide(Od.loc[Oa.group].reset_index(), False)
    Oa['p_anem_doc'] = 1 - np.stack(Od.loc[Oa.group].P.values)[:, 0]
    Oa['err_doc'] = E.expected_error(Od.loc[Oa.group].reset_index())
    W = Oa.sort_values('err', ascending=False).head(a.top).merge(info.drop(columns=['y', 'hb_g_l']), on='group')
    W['правда'] = [E.CLASSES[i] for i in W.y]
    W['ответ авто'] = [E.CLASSES[i] for i in W.k]
    W['ответ врач-маска'] = [E.CLASSES[i] for i in W.k_doc]
    cols = ['group', 'country', 'sex', 'age', 'hb_g_l', 'правда', 'ответ авто', 'p_anem', 'ответ врач-маска',
            'p_anem_doc', 'err', 'err_doc', 'iou_auto', 'iou_v11', 'conj_frac_doc', 'conj_frac_auto', 'frame_L',
            'sclera_found', 'blur', 'issues']
    W[cols].to_csv(os.path.join(out, 'worst.csv'), index=False, float_format='%.3f')
    sheet = []
    for i, r in W.iterrows():
        p = pics[r.group]
        img = E.panel(p['crop'], [p['doctor'], p['auto'], p['v11']],
                      [f'#{i + 1} Hb {r.hb_g_l:.0f} y={r.y} auto={r.k} doc={r.k_doc}', 'doctor',
                       f'auto v12 IoU {r.iou_auto:.2f}', f'v11 IoU {r.iou_v11:.2f}'])
        cv2.imwrite(os.path.join(out, 'worst', f'{i + 1:02d}_{r.group}.png'), img[..., ::-1])
        sheet.append(cv2.resize(img, (img.shape[1] * 200 // img.shape[0], 200)))
    w = max(s.shape[1] for s in sheet)
    sheet = np.vstack([np.pad(s, ((0, 0), (0, w - s.shape[1]), (0, 0)), constant_values=255) for s in sheet])
    cv2.imwrite(os.path.join(out, 'worst_sheet.jpg'), sheet[..., ::-1], [cv2.IMWRITE_JPEG_QUALITY, 85])

    # ------------------------------------------------ 4. разбивка (conj, общие, порог 0.80)
    meta = info.copy()
    meta['возраст'] = pd.cut(meta.age, [0, 30, 50, 65, 120], labels=['<30', '30–49', '50–64', '65+']).astype(str)
    meta['яркость кадра L*'] = E.tertile_labels(meta.frame_L, 'L*')
    meta['IoU авто'] = E.tertile_labels(meta.iou_auto, 'IoU', '{:.2f}')
    meta['качество'] = np.where(meta.issues.str.len() > 0, 'есть замечания', 'без замечаний')
    meta['склера найдена'] = np.where(meta.sclera_found, 'да', 'нет')
    Os = {'врач': oof[('conj', 'doctor', 'общие')], 'авто': oof[('conj', 'auto', 'общие')]}
    B = pd.concat([E.breakdown(Os, meta.rename(columns={'country': 'страна', 'sex': 'пол'}), by)
                   for by in ('страна', 'пол', 'возраст', 'яркость кадра L*', 'IoU авто', 'качество',
                              'склера найдена')])
    B.to_csv(os.path.join(out, 'breakdown.csv'), index=False, float_format='%.3f')
    Bv = B.assign(**{c: B[c].map(fmt_pct) for c in ('bal_acc', 'sens', 'spec')}, auc=B.auc.map(
        lambda v: '—' if v != v else f'{v:.3f}'))
    info.to_csv(os.path.join(out, 'per_patient.csv'), index=False, float_format='%.3f')

    rep = [f'# Анализ ошибок v12 · {a.tag}', '',
           (f'Eyes-defy, {len(info)} взрослых (исключены из-за испорченной маски врача: {sorted(bad) or "нет"}), '
            'фолды манифеста по пациенту. IoU с маской врача: '
            f'v12 авто среднее {info.iou_auto.mean():.2f} (медиана {info.iou_auto.median():.2f}), '
            f'v11 {info.iou_v11.mean():.2f} (медиана {info.iou_v11.median():.2f}).'), '',
           '## Маска врача vs автоматические', '', md_table(T), '',
           '## Разница «маска врача − авто» (одни пациенты, 95% ДИ бутстрэп)', '', md_table(D), '',
           '## Матрицы ошибок (конъюнктива, общие пациенты)', '']
    for title, C in conf:
        rep += [f'**{title}**', '', md_table(C.reset_index().rename(columns={'index': ''})), '']
    rep += [f'## {a.top} худших (авто-маска v12, конъюнктива, argmax; ошибка = Σ P·|класс − правда|)', '',
            md_table(W[cols].round(3), '{:.2f}'), '',
            '## Разбивка (конъюнктива, порог 0.80)', '',
            md_table(Bv[['признак', 'группа', 'маска', 'n', 'bal_acc', 'auc', 'sens', 'spec', 'примечание']]), '']
    with open(os.path.join(out, 'report.md'), 'w') as f:
        f.write('\n'.join(rep))
    print('\n'.join(rep[:12 + len(conf) * 4]))
    print(f'\nОтчёт: {os.path.join(out, "report.md")}')


if __name__ == '__main__':
    main()
