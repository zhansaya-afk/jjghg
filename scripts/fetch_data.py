"""Скачивает и раскладывает датасеты в data/ (запускается в Colab: из облака Claude Kaggle закрыт).

    python scripts/fetch_data.py --eyes-defy                         # Kaggle harshwardhanfartale/eyes-defy-anemia
    python scripts/fetch_data.py --external                          # внешние наборы без Hb (anemia/external.py)

Результат: data/eyes_defy_anemia/{raw/, labels.csv}; data/external/{<набор>/raw/, index.csv}. CP-AnemiC исключён из проекта 2026-10-19 и не скачивается.
Исходные архивы не изменяются. Каждый файл сверяется по sha1 со списком scripts/eyes_defy.sha1 (только имена
и хэши, без меток): несовпадение — понятная ошибка, а не тихо другие данные.
"""
import argparse
import glob
import hashlib
import os
import re
import shutil
import zipfile

import pandas as pd


HERE = os.path.dirname(os.path.abspath(__file__))
TIMEOUT = 600            # с на одну загрузку: сеть пропала — ошибка, а не вечное ожидание


def read_sha1_list(name):
    """scripts/<name>: строки «sha1 путь» или «имя sha1» → {путь: sha1}; # — комментарии."""
    out = {}
    for line in open(os.path.join(HERE, name), encoding='utf-8'):
        line = line.strip()
        if line and not line.startswith('#'):
            a, b = line.split(' ', 1)
            path, sha = (b, a) if re.fullmatch(r'[0-9a-f]{40}', a) else (a, b)
            out[path] = sha
    return out


def sha1_file(path):
    h = hashlib.sha1()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def verify_files(root, expected, what):
    """Все файлы из списка есть в root и совпадают по sha1, иначе RuntimeError с первыми расхождениями."""
    bad = [p for p, s in expected.items()
           if not os.path.isfile(os.path.join(root, p)) or sha1_file(os.path.join(root, p)) != s]
    if bad:
        raise RuntimeError(f'{what}: {len(bad)} из {len(expected)} файлов нет или они другие (sha1), например '
                           f'{bad[:3]} — источник изменился; данные не используются')
    print(f'{what}: целостность проверена, {len(expected)} файлов совпали по sha1')


def _download(url, path):
    import urllib.request
    if not os.path.exists(path):
        with urllib.request.urlopen(url, timeout=TIMEOUT) as r, open(path + '.part', 'wb') as f:
            shutil.copyfileobj(r, f, 1 << 20)
        os.replace(path + '.part', path)
    return path


# ------------------------------------------------------------------ Eyes-defy-anemia
def _find_col(cols, pats):
    for c in cols:
        if any(re.search(p, str(c).strip().lower()) for p in pats):
            return c
    return None


def _norm_id(x):
    """«007», «7.0», «Patient 7» → «7»: в таблицах и в именах папок номера записаны по-разному."""
    m = re.search(r'\d+', str(x))
    return str(int(m.group())) if m else None


MASK_ROLES = (('forniceal_palpebral', r'forn\w*ceal_pa\w*ebral'), ('forniceal', r'forn\w*ceal'),
              ('palpebral', r'pa\w*ebral'))


def mask_role(name):
    """Роль файла маски по имени; None — не маска (исходное фото). Регулярки, а не подстрока «palpebral»: в
    архиве есть опечатки — India/7 «…_papebral.png», Italy/95 «…_palplebral.png» (аудит данных 2026-10-19)."""
    n = name.lower()
    for r, pat in MASK_ROLES:
        if re.search(pat, n):
            return r
    return None


def eyes_defy_labels(raw):
    """Таблицы Hb (xlsx Италия/Индия) + папки пациентов с фото и масками врачей."""
    recs = []
    for t in glob.glob(f'{raw}/**/*', recursive=True):
        if not t.lower().endswith(('.xlsx', '.xls')):
            continue
        country = 'India' if 'india' in t.lower() else 'Italy' if 'italy' in t.lower() else 'NA'
        for _, df in pd.read_excel(t, sheet_name=None).items():
            pid = _find_col(df.columns, [r'^n', r'number', r'^id', r'patient'])
            hb = _find_col(df.columns, [r'^hg?b', r'hemoglob', r'haemoglob', r'emoglob'])
            sex = _find_col(df.columns, [r'^sex', r'gender'])
            age = _find_col(df.columns, [r'^age', r'^eta'])
            if pid is None or hb is None:
                continue
            for _, r in df.iterrows():
                hbv = pd.to_numeric(str(r[hb]).replace(',', '.'), errors='coerce')
                if pd.isna(hbv):
                    continue
                recs.append({'country': country, 'pid': _norm_id(r[pid]), 'hb_g_dl': float(hbv),
                             'sex': r[sex] if sex else None, 'age': r[age] if age else None})
    tab = pd.DataFrame(recs).dropna(subset=['pid']).drop_duplicates(['country', 'pid'])
    # папка пациента = любая папка, где есть исходное фото и маска (как в рабочем загрузчике v10)
    folders = {}
    for f in glob.glob(f'{raw}/**/*', recursive=True):
        if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')):
            folders.setdefault(os.path.dirname(f), []).append(os.path.basename(f))
    rows = []
    for d, imgs in sorted(folders.items()):
        # «(1)»-копии — в конец (в India/8 они побайтно равны основным файлам)
        low = {f: f.lower() for f in sorted(imgs, key=lambda f: ('(' in f, f))}
        role = {f: mask_role(n) for f, n in low.items()}
        palp = [f for f in low if role[f] == 'palpebral']
        forn = [f for f in low if role[f] == 'forniceal']
        both = [f for f in low if role[f] == 'forniceal_palpebral']
        orig = [f for f in low if role[f] is None]
        if not orig or not (palp or both):
            continue
        rel = lambda fs: os.path.relpath(os.path.join(d, fs[0]), raw) if fs else None
        path = d.lower()
        rows.append({'country': 'India' if 'india' in path else 'Italy' if 'italy' in path else 'NA',
                     'pid': _norm_id(os.path.basename(d)), 'image': rel(orig),
                     'mask_palpebral': rel(palp or both),     # нет отдельной маски века → общая (как в v10)
                     'mask_forniceal': rel(forn), 'mask_forniceal_palpebral': rel(both)})
    if (tab.country == 'NA').all():                           # страна не определяется по таблицам
        tab = tab.drop(columns='country').drop_duplicates('pid')
        return pd.DataFrame(rows).merge(tab, on='pid', how='left')
    return pd.DataFrame(rows).merge(tab, on=['country', 'pid'], how='left')


ED_URL = 'https://www.kaggle.com/api/v1/datasets/download/harshwardhanfartale/eyes-defy-anemia'


def _download_eyes_defy(out):
    """kagglehub (api.kaggle.com), иначе прямой zip публичного датасета (www.kaggle.com → storage.googleapis.com)."""
    try:
        import kagglehub
        return kagglehub.dataset_download('harshwardhanfartale/eyes-defy-anemia')
    except Exception as e:                                    # нет kagglehub или api.kaggle.com закрыт прокси
        print(f'kagglehub недоступен ({type(e).__name__}), качаю zip напрямую')
    return _download(ED_URL, os.path.join(out, 'eyes_defy_anemia.zip'))


def fetch_eyes_defy(out, zip_path=None):
    src = zip_path or _download_eyes_defy(out)
    raw = os.path.join(out, 'eyes_defy_anemia', 'raw')
    if os.path.isfile(src):
        zipfile.ZipFile(src).extractall(raw)
    else:
        shutil.copytree(src, raw, dirs_exist_ok=True)
    verify_files(raw, read_sha1_list('eyes_defy.sha1'), 'Eyes-defy-anemia')
    write_eyes_defy_labels(out)


def write_eyes_defy_labels(out):
    """labels.csv из уже распакованного raw/ (без скачивания, ~1 с): ячейка 2 блокнота пересобирает его каждый раз,
    чтобы исправления разбора папок доходили до манифеста без повторной загрузки."""
    raw = os.path.join(out, 'eyes_defy_anemia', 'raw')
    lab = eyes_defy_labels(raw)
    lab.to_csv(os.path.join(out, 'eyes_defy_anemia', 'labels.csv'), index=False)
    print(f'Eyes-defy-anemia: {len(lab)} пациентов, с Hb {lab.hb_g_dl.notna().sum()}, '
          f'с маской {lab.mask_palpebral.notna().sum()}; {lab.country.value_counts().to_dict()}')
    return lab


# ------------------------------------------------------------------ внешние наборы (без Hb, только проверка)
def fetch_external(out, keys=None, zips=None):
    """Наборы anemia.external.SOURCES → <out>/external/<key>/raw + <out>/external/index.csv. Копии Eyes-defy
    (anemia.external.COPIES) не скачиваются; файлы, совпавшие по sha1 с Eyes-defy, помечаются ed_copy."""
    import sys
    sys.path.insert(0, os.path.dirname(HERE))
    from anemia import external as EXT
    keys = keys or list(EXT.SOURCES)
    for ref, why in EXT.COPIES.items():
        print(f'пропущен {ref}: {why}')
    os.makedirs(os.path.join(out, 'external'), exist_ok=True)
    for key in keys:
        raw = os.path.join(out, 'external', key, 'raw')
        if not os.path.isdir(raw):
            src = (zips or {}).get(key) or _download(EXT.kaggle_url(EXT.SOURCES[key]['ref']),
                                                     os.path.join(out, 'external', f'{key}.zip'))
            tmp = raw + '.part'
            shutil.rmtree(tmp, ignore_errors=True)
            zipfile.ZipFile(src).extractall(tmp)          # прерванная распаковка не оставит «готовую» папку raw
            os.replace(tmp, raw)
    ed = read_sha1_list('eyes_defy.sha1').values()
    ed_raw = os.path.join(out, 'eyes_defy_anemia', 'raw')
    ed_photos = [f for f in glob.glob(os.path.join(ed_raw, '**', '*.jpg'), recursive=True)]
    if ed_photos:
        print(f'сверка с Eyes-defy по миниатюрам: {len(ed_photos)} фото...', flush=True)
    else:
        print('Eyes-defy не скачан: копии ищутся только по sha1 (без сверки миниатюр)')
    idx = EXT.build_index(out, keys, ed, [EXT.thumb(f) for f in ed_photos])
    path = os.path.join(out, 'external', 'index.csv')
    idx.to_csv(path, index=False)
    print(f'Внешние наборы → {path}\n{EXT.summary(idx).to_string()}')
    return idx


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.environ.get('ANEMIA_DATA', 'data'))
    ap.add_argument('--eyes-defy', action='store_true')
    ap.add_argument('--eyes-defy-zip', help='уже скачанный zip Eyes-defy (вместо Kaggle)')
    ap.add_argument('--relabel', action='store_true',
                    help='пересобрать eyes_defy_anemia/labels.csv из уже распакованных файлов (без скачивания)')
    ap.add_argument('--external', nargs='*', metavar='KEY',
                    help='внешние наборы без Hb (anemia/external.py SOURCES); без ключей — все')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.eyes_defy or a.eyes_defy_zip:
        fetch_eyes_defy(a.out, a.eyes_defy_zip)
    elif a.relabel:
        write_eyes_defy_labels(a.out)
    if a.external is not None:
        fetch_external(a.out, a.external)

