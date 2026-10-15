"""Скачивает и раскладывает датасеты в data/ (запускается в Colab: из облака Claude Kaggle закрыт).

    python scripts/fetch_data.py --eyes-defy                         # Kaggle harshwardhanfartale/eyes-defy-anemia
    python scripts/fetch_data.py --cp-zip /content/drive/MyDrive/anemia/CP-AnemiC_dataset.zip

Результат: data/eyes_defy_anemia/{raw/, labels.csv}, data/cp_anemic/{images/, labels.csv}.
Исходные архивы не изменяются.
"""
import argparse
import glob
import io
import os
import re
import shutil
import zipfile

import pandas as pd


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
        low = {f: f.lower() for f in sorted(imgs)}
        palp = [f for f, n in low.items() if 'palpebral' in n and 'forniceal' not in n]
        forn = [f for f, n in low.items() if 'forniceal' in n and 'palpebral' not in n]
        both = [f for f, n in low.items() if 'palpebral' in n and 'forniceal' in n]
        orig = [f for f, n in low.items() if 'palpebral' not in n and 'forniceal' not in n]
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
    import urllib.request
    zpath = os.path.join(out, 'eyes_defy_anemia.zip')
    if not os.path.exists(zpath):
        urllib.request.urlretrieve(ED_URL, zpath + '.part')
        os.replace(zpath + '.part', zpath)
    return zpath


def fetch_eyes_defy(out, zip_path=None):
    src = zip_path or _download_eyes_defy(out)
    raw = os.path.join(out, 'eyes_defy_anemia', 'raw')
    if os.path.isfile(src):
        zipfile.ZipFile(src).extractall(raw)
    else:
        shutil.copytree(src, raw, dirs_exist_ok=True)
    lab = eyes_defy_labels(raw)
    lab.to_csv(os.path.join(out, 'eyes_defy_anemia', 'labels.csv'), index=False)
    print(f'Eyes-defy-anemia: {len(lab)} пациентов, с Hb {lab.hb_g_dl.notna().sum()}, '
          f'с маской {lab.mask_palpebral.notna().sum()}; {lab.country.value_counts().to_dict()}')


# ------------------------------------------------------------------ CP-AnemiC
def _open_nested(zf):
    """Архив Mendeley — zip внутри zip: спускаемся, пока не найдём xlsx."""
    while not any(n.lower().endswith('.xlsx') for n in zf.namelist()):
        inner = [n for n in zf.namelist() if n.lower().endswith('.zip')]
        if not inner:
            raise ValueError('в архиве нет xlsx CP-AnemiC')
        zf = zipfile.ZipFile(io.BytesIO(zf.read(inner[0])))
    return zf


def fetch_cp_anemic(zip_path, out):
    zf = _open_nested(zipfile.ZipFile(zip_path))
    d = os.path.join(out, 'cp_anemic')
    os.makedirs(os.path.join(d, 'images'), exist_ok=True)
    xlsx = next(n for n in zf.namelist() if n.lower().endswith('.xlsx'))
    sheets = pd.read_excel(io.BytesIO(zf.read(xlsx)), sheet_name=None)
    tab = next(v for v in sheets.values() if {'File_Name', 'Hb_g_dl'} <= set(v.columns))
    pngs = {os.path.basename(n): n for n in zf.namelist() if '/All Categories/' in n and n.endswith('.png')}
    rows = []
    for _, r in tab.iterrows():
        src = pngs.get(r.File_Name)
        if src is None:
            continue
        with open(os.path.join(d, 'images', r.File_Name), 'wb') as f:
            f.write(zf.read(src))
        rows.append({'file': f'images/{r.File_Name}', 'hb_g_dl': float(r.Hb_g_dl), 'label': r.Folder_Label})
    pd.DataFrame(rows).to_csv(os.path.join(d, 'labels.csv'), index=False)
    print(f'CP-AnemiC: {len(rows)} фото')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.environ.get('ANEMIA_DATA', 'data'))
    ap.add_argument('--eyes-defy', action='store_true')
    ap.add_argument('--eyes-defy-zip', help='уже скачанный zip Eyes-defy (вместо Kaggle)')
    ap.add_argument('--cp-zip')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.eyes_defy or a.eyes_defy_zip:
        fetch_eyes_defy(a.out, a.eyes_defy_zip)
    if a.cp_zip:
        fetch_cp_anemic(a.cp_zip, a.out)
