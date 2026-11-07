#!/usr/bin/env python
"""Проверка интеграции main + claude/project-thread-fpcxzb (временная ветка claude/integration-check-fpcxzb, не для main).

Шаги (ничего не обучает для production, файлы моделей не пишет и не меняет):
  1. QC на всех снимках Eyes-defy через production-путь predict.analyze_photo. Сегментатор века каждого снимка —
     из evaluate.conj_segmenters, обученный БЕЗ фолда этого снимка (как маски тестовых снимков в evaluate), в память,
     на диск не сохраняется. Модель риска не нужна: заглушка останавливает analyze_photo сразу после QC и выбора
     модели (до признаков, OOD и риска). Рядом — маршрут evaluate (eye_route) на том же снимке и сегментаторе:
     снимок, который evaluate принимает, а analyze_photo отклоняет, — кандидат в необоснованные отказы.
  2. scripts/evaluate.py --task core --no-history на этой (интегрированной) версии.
  3. То же на origin/main (git worktree рядом с репозиторием), те же данные.
  4. Сводка: QC (прошли / отказы / причины) и метрики main vs интеграция.
Результаты: $ANEMIA_BASE/integration_check/ (вне git). В репозиторий ничего не пишется.
"""
import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import evaluate as EV  # noqa: E402
from anemia import predict as PR  # noqa: E402


class PassedQC(Exception):
    """analyze_photo дошёл до признаков модели key — QC пройден."""


class StubModel(dict):
    """Модель-заглушка: даёт только название; первое обращение к признакам → PassedQC(key)."""

    def __init__(self, key):
        super().__init__()
        self.key = key

    def __getitem__(self, k):
        if k == 'title':
            return f'заглушка {self.key}'
        raise PassedQC(self.key)

    def get(self, k, default=None):
        return self[k]


def qc_one(path, row, seg, lm):
    bundle = {'models': {k: StubModel(k) for k in ('eyes_defy', 'eyes_defy_conj', 'eyes_defy_sclera')},
              'conj_seg': seg}
    sex = row['sex'] if isinstance(row['sex'], str) else None
    age = None if pd.isna(row['age']) else float(row['age'])
    try:
        r = PR.analyze_photo(path, sex=sex, age=age, pregnant=None, bundle=bundle, landmarker=lm)
        out = {'qc': 'отказ', 'stage': r['stage'], 'model': '', 'reason': '; '.join(map(str, r['issues']))[:300],
               'tissue_status': json.dumps(r.get('tissue_status') or {}, ensure_ascii=False)}
    except PassedQC as e:
        out = {'qc': 'прошёл', 'stage': 'passed_qc', 'model': e.args[0], 'reason': '', 'tissue_status': ''}
    det, _ = EV._det_compute(path)
    _, key, why = EV.eye_route(det, seg)
    out.update(eval_route=key or 'отказ', eval_why=(why or '')[:200])
    return out


def reason_class(s):
    s = s.lower()
    for pat, name in (('пикс', 'мало пикселей века (исходные)'), ('резк|размыт', 'размыто'), ('jpeg|сжат|блок', 'сжатие JPEG'),
                      ('шум', 'шум'), ('темно', 'темно'), ('пересвет|вспышк', 'пересвет'),
                      ('анатом', 'анатомия века'), ('не найден', 'ткань не найдена'), ('мелк', 'ткань мелкая'),
                      ('вырезан', 'вырезанная ткань')):
        if re.search(pat, s):
            return name
    return 'другое'


def run_qc(data, out_dir, jobs):
    man = pd.read_csv(os.path.join(data, 'manifest.csv'))
    df = man[man.dataset == 'eyes_defy'].reset_index(drop=True)
    print(f'QC: {len(df)} снимков Eyes-defy; обучаю сегментаторы века по фолдам (без фолда снимка)...', flush=True)
    t0 = time.time()
    segs = EV.conj_segmenters(data, df, jobs, full=False, external=True)
    print(f'  сегментаторы: {sorted(segs)} за {time.time() - t0:.0f} с', flush=True)
    lm, note = PR.face_landmarker()
    if lm is None:
        print('  MediaPipe недоступен:', note, '— поиск лица выключен (как в Colab без mediapipe)')
    recs = []
    for i, row in enumerate(df.to_dict('records')):
        res = qc_one(os.path.join(data, row['image']), row, segs[(int(row['fold']),)], lm)
        recs.append({'patient_id': row['patient_id'], 'fold': int(row['fold']), **res})
        if (i + 1) % 25 == 0:
            print(f'  {i + 1}/{len(df)}', flush=True)
    Q = pd.DataFrame(recs)
    Q['reason_class'] = [reason_class(r) if q == 'отказ' else '' for q, r in zip(Q.qc, Q.reason)]
    Q.to_csv(os.path.join(out_dir, 'qc_per_image.csv'), index=False)
    return Q


def run_evaluate(repo, data, tag, log):
    cmd = [sys.executable, os.path.join(repo, 'scripts', 'evaluate.py'), '--task', 'core', '--dataset', 'eyes_defy',
           '--data', data, '--tag', tag, '--no-history']
    print(f'evaluate ({tag}) в {repo} ... лог: {log}', flush=True)
    t0 = time.time()
    with open(log, 'w') as f:
        rc = subprocess.run(cmd, cwd=repo, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f'  код выхода {rc}, {time.time() - t0:.0f} с', flush=True)
    text = open(log).read()
    m = [ln.split(': ', 1)[1].strip() for ln in text.splitlines() if ln.startswith('Результаты запуска: ')]
    run_dir = m[-1] if m else None
    metrics = pd.read_csv(os.path.join(run_dir, 'metrics.csv')) if run_dir and os.path.exists(
        os.path.join(run_dir, 'metrics.csv')) else None
    return rc, text, metrics


def tables(text):
    """Таблицы evaluate (строки «|…») и их подписи, без таблицы времени этапов."""
    out, skip = [], False
    for line in text.splitlines():
        if line.startswith('|'):
            if line.startswith('| Этап'):
                skip = True
            if not skip:
                out.append(line)
            continue
        skip = False
        if line.startswith(('Калибровка', 'eyes_defy')):
            out.append('')
            out.append(line)
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default=os.environ.get('ANEMIA_DATA', 'data'))
    ap.add_argument('--out', default=os.path.join(os.environ.get('ANEMIA_BASE', '.'), 'integration_check'))
    ap.add_argument('--jobs', type=int, default=-1)
    ap.add_argument('--skip-qc', action='store_true')
    ap.add_argument('--skip-eval', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    head = subprocess.run(['git', '-C', ROOT, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
    print(f'Интеграция: коммит {head}; данные {a.data}; результаты {a.out}', flush=True)

    if not a.skip_qc:
        Q = run_qc(a.data, a.out, a.jobs)
        n = len(Q)
        ok = int((Q.qc == 'прошёл').sum())
        print('\n=== QC через analyze_photo (сегментатор без фолда снимка) ===')
        print(f'Снимков {n}: прошли QC {ok} ({ok / n:.1%}), отказ {n - ok} ({(n - ok) / n:.1%})')
        print('Модель после QC:', dict(collections.Counter(Q.model[Q.qc == 'прошёл'])))
        print('Отказы по этапу:', dict(collections.Counter(Q.stage[Q.qc == 'отказ'])))
        print('Отказы по причине:', dict(collections.Counter(Q.reason_class[Q.qc == 'отказ'])))
        both = pd.crosstab(Q.qc, Q.eval_route.ne('отказ').map({True: 'evaluate принимает', False: 'evaluate отказ'}))
        print('analyze_photo × маршрут evaluate:\n' + both.to_string())
        sus = Q[(Q.qc == 'отказ') & (Q.eval_route != 'отказ')]
        print(f'Кандидаты в необоснованные отказы (evaluate принимает, analyze_photo отклоняет): {len(sus)}')
        for r in sus.itertuples():
            print(f'  {r.patient_id} (фолд {r.fold}): {r.stage} — {r.reason[:160]}')
        extra = Q[(Q.qc == 'прошёл') & (Q.eval_route == 'отказ')]
        print(f'Наоборот (evaluate отклоняет, analyze_photo принимает): {len(extra)}')

    if not a.skip_eval:
        main_dir = os.path.join(os.path.dirname(ROOT), 'main_wt')
        subprocess.run(['git', '-C', ROOT, 'fetch', '-q', 'origin', 'main'], check=True)
        if not os.path.isdir(main_dir):
            subprocess.run(['git', '-C', ROOT, 'worktree', 'add', '--detach', main_dir, 'origin/main'], check=True)
        else:
            subprocess.run(['git', '-C', main_dir, 'checkout', '-q', '--detach', 'origin/main'], check=True)
        mh = subprocess.run(['git', '-C', main_dir, 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True).stdout.strip()
        res = {}
        for name, repo in (('main ' + mh, main_dir), ('интеграция ' + head, ROOT)):
            res[name] = run_evaluate(repo, a.data, 'integration_check', os.path.join(a.out, f'evaluate_{name.split()[0]}.log'))
        print('\n=== evaluate --task core: main vs интеграция ===')
        cols = ['dataset', 'method', 'n', 'n_accepted', 'n_ood', 'auc', 'auc_lo', 'auc_hi', 'sens', 'sens_lo', 'sens_hi',
                'spec', 'spec_lo', 'spec_hi', 'bacc2', 'mae', 'light_med', 'iou_mean']
        for name, (rc, text, M) in res.items():
            print(f'\n--- {name} (код выхода {rc}) ---')
            if M is not None:
                print(M[[c for c in cols if c in M.columns]].to_string(index=False))
            print(tables(text))
    print(f'\nГотово. Файлы: {a.out}')


if __name__ == '__main__':
    main()
