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
                recs.append({'country': country, 'pid': str(r[pid]).split('.')[0].strip(), 'hb_g_dl': float(hbv),
                             'sex': r[sex] if sex else None, 'age': r[age] if age else None})
    tab = pd.DataFrame(recs).drop_duplicates(['country', 'pid'])
    rows = []
    for d in sorted(glob.glob(f'{raw}/**/', recursive=True)):
        name = os.path.basename(os.path.normpath(d))
        country = 'India' if '/india/' in d.lower() else 'Italy' if '/italy/' in d.lower() else None
        if country is None or not name.isdigit():
            continue
        imgs = [f for f in os.listdir(d) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        raw_img = [f for f in imgs if not re.search(r'forniceal|palpebral', f.lower())]
        mask = lambda k: next((os.path.relpath(os.path.join(d, f), raw) for f in imgs
                               if f.lower().endswith(f'{k}.png')), None)
        rows.append({'country': country, 'pid': name,
                     'image': os.path.relpath(os.path.join(d, raw_img[0]), raw) if raw_img else None,
                     'mask_palpebral': mask('_palpebral'), 'mask_forniceal': mask('_forniceal'),
                     'mask_forniceal_palpebral': mask('_forniceal_palpebral')})
    return pd.DataFrame(rows).merge(tab, on=['country', 'pid'], how='left')


def fetch_eyes_defy(out):
    import kagglehub
    src = kagglehub.dataset_download('harshwardhanfartale/eyes-defy-anemia')
    raw = os.path.join(out, 'eyes_defy_anemia', 'raw')
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
    ap.add_argument('--cp-zip')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.eyes_defy:
        fetch_eyes_defy(a.out)
    if a.cp_zip:
        fetch_cp_anemic(a.cp_zip, a.out)
