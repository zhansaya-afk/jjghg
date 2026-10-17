"""Запись результатов запусков без перезаписи: уникальный ID, папка запуска, история метрик с резервной копией.

<data>/results/runs/<run_id>/ — run.json (код, параметры, время, тайминги, статус), metrics.csv, OOF, лог.
Папка создаётся только новой (exist_ok=False): результаты прошлого запуска перезаписать нельзя.
"""
import datetime as dt
import glob
import json
import os
import re
import secrets
import shutil

import pandas as pd


def new_run_id(tag, now=None):
    now = now or dt.datetime.now()
    safe = re.sub(r'[^A-Za-z0-9_.-]+', '-', str(tag)).strip('-') or 'run'
    return f'{now:%Y%m%d-%H%M%S}-{safe}-{secrets.token_hex(3)}'


def create_run_dir(base, tag):
    os.makedirs(base, exist_ok=True)
    for _ in range(10):
        run_id = new_run_id(tag)
        path = os.path.join(base, run_id)
        try:
            os.makedirs(path, exist_ok=False)
            return run_id, path
        except FileExistsError:
            continue
    raise RuntimeError('не удалось создать уникальную папку запуска')


def write_json(path, obj):
    tmp = path + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, default=str)
    os.replace(tmp, path)


def backup_file(path, backup_dir):
    """Копия файла в backup_dir с отметкой времени; существующие копии не трогаются."""
    os.makedirs(backup_dir, exist_ok=True)
    stem, ext = os.path.splitext(os.path.basename(path))
    dst = os.path.join(backup_dir, f'{stem}.{dt.datetime.now():%Y%m%d-%H%M%S}.{secrets.token_hex(2)}{ext}')
    shutil.copy2(path, dst)
    return dst


def append_history(path, rows, cols, fixed, backup_dir):
    """Дописать строки в CSV истории. Если колонки файла другие — сначала резервная копия, потом новый заголовок
    (старые строки сохраняются, новые колонки у них пустые)."""
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    if os.path.exists(path):
        old = pd.read_csv(path)
        if list(old.columns) != cols:
            backup_file(path, backup_dir)
            old.reindex(columns=cols).to_csv(path, index=False, float_format='%.4f')
    fixed = {k: v for k, v in fixed.items() if k in cols}
    H = pd.DataFrame([{**dict.fromkeys(cols), **fixed, **{k: v for k, v in r.items() if k in cols}}
                      for r in rows])[cols]
    H.to_csv(path, mode='a', header=not os.path.exists(path), index=False, float_format='%.4f')


def preserve_before_overwrite(path, runs_base, backup_dir):
    """Перед перезаписью файла «последнего прогона» (oof_<tag>_*.csv): если такой же копии нет ни в одной папке
    запуска (файл от старой версии скрипта), сохранить его в backup_dir. → путь копии или None."""
    if not os.path.exists(path):
        return None
    with open(path, 'rb') as f:
        cur = f.read()
    for q in glob.glob(os.path.join(runs_base, '*', os.path.basename(path))):
        with open(q, 'rb') as f:
            if f.read() == cur:
                return None
    return backup_file(path, backup_dir)
