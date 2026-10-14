"""Смоук-тест: evaluate.py проходит от манифеста до таблицы на синтетике. Цифры синтетики не являются точностью."""
import os
import subprocess
import sys

import numpy as np
import pandas as pd
from PIL import Image

from anemia.data import build_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_evaluate_runs_on_synthetic(tmp_path):
    rng = np.random.default_rng(0)
    rows = []
    os.makedirs(tmp_path / 'phone', exist_ok=True)
    for p in range(30):
        hb = rng.uniform(80, 150)
        pale = (150 - hb) / 70
        base = np.array([0.75, 0.30 + 0.25 * pale, 0.30 + 0.2 * pale])
        img = np.clip(base + rng.normal(0, 0.03, (40, 50, 3)), 0, 1)
        Image.fromarray((img * 255).astype(np.uint8)).save(tmp_path / 'phone' / f'p{p}.png')
        rows.append({'file': f'p{p}.png', 'patient_id': p, 'hb_g_l': hb, 'sex': 'F', 'age': 30})
    pd.DataFrame(rows).to_csv(tmp_path / 'phone' / 'hb.csv', index=False)
    build_manifest(str(tmp_path), n_folds=3).to_csv(tmp_path / 'manifest.csv', index=False)
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'evaluate.py'), '--data', str(tmp_path),
                        '--model', 'ridge', '--no-history', '--tag', 'pytest', '--jobs', '1'],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert '| phone | ridge · маска full | 30 |' in r.stdout
    assert 'MAE константы' in r.stdout
    # было/стало: второй прогон с другим числом копий света сравнивается с первым по OOF
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'evaluate.py'), '--data', str(tmp_path),
                        '--model', 'ridge', '--no-history', '--tag', 'pytest2', '--jobs', '1', '--relight', '1',
                        '--compare', 'pytest'], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert 'Было/стало относительно «pytest»' in r.stdout
    assert any(v in r.stdout for v in ('лучше', 'хуже', 'в пределах шума', 'смешанно'))


def test_eyes_defy_layout_and_seg_baseline(tmp_path):
    """Раскладка как у Eyes-defy (папка пациента: фото + *_palpebral.png + *_forniceal_palpebral.png)
    → fetch-разметка → манифест → evaluate --task seg. Синтетика: проверяется только, что код проходит."""
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    from fetch_data import eyes_defy_labels
    raw = tmp_path / 'eyes_defy_anemia' / 'raw'
    rng = np.random.default_rng(0)
    recs = []
    for country in ('Italy', 'India'):
        for p in range(1, 11):
            d = raw / 'dataset' / country / str(p)
            os.makedirs(d, exist_ok=True)
            img = np.full((120, 160, 3), 200, np.uint8)
            img[40:70, 50:110] = (40, 30, 30)                       # «радужка»
            img[80:105, 30 + p:130 - p] = (190, 70 + 3 * p, 70)      # «конъюнктива»
            img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
            Image.fromarray(img).save(d / f'{p}.jpg')
            m = np.zeros_like(img); m[80:105, 30 + p:130 - p] = img[80:105, 30 + p:130 - p]
            Image.fromarray(m).save(d / f'{p}_palpebral.png')
            Image.fromarray(np.zeros_like(img)).save(d / f'{p}_forniceal_palpebral.png')
            recs.append({'Number': f'{p:03d}', 'Hgb': str(9 + p * 0.5).replace('.', ','), 'Sex': 'F' if p % 2 else 'M'})
        pd.DataFrame(recs[-10:]).to_excel(raw / 'dataset' / country / f'{country}.xlsx', index=False)
    lab = eyes_defy_labels(str(raw))
    assert len(lab) == 20 and lab.hb_g_dl.notna().all()
    assert lab.mask_palpebral.str.endswith('_palpebral.png').all()
    assert not lab.mask_palpebral.str.contains('forniceal').any()
    lab.to_csv(tmp_path / 'eyes_defy_anemia' / 'labels.csv', index=False)
    build_manifest(str(tmp_path), n_folds=3).to_csv(tmp_path / 'manifest.csv', index=False)
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'evaluate.py'), '--data', str(tmp_path),
                        '--model', 'ridge', '--no-history', '--tag', 'pytest', '--jobs', '1'],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-2000:]
    assert '| eyes_defy | v11 pixel-HGB | 20 |' in r.stdout
    assert 'ridge · маска doctor' in r.stdout and 'ridge · маска pred' in r.stdout


def test_history_gains_new_columns(tmp_path, monkeypatch):
    """Старый metrics_history.csv (меньше колонок) переписывается с новой шапкой, строки сохраняются."""
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import evaluate
    hist = tmp_path / 'h.csv'
    old_cols = evaluate.HIST_COLS[:evaluate.HIST_COLS.index('found') + 1]
    pd.DataFrame([{c: '' for c in old_cols} | {'tag': 'old', 'mae': '16.4'}])[old_cols].to_csv(hist, index=False)
    monkeypatch.setattr(evaluate, 'HISTORY', str(hist))
    evaluate.append_history([{'task': 'hb', 'mae': 15.7, 'mae_null': 17.0, 'verdict': 'лучше'}], 'new', 'abc')
    H = pd.read_csv(hist)
    assert list(H.columns) == evaluate.HIST_COLS
    assert H.tag.tolist() == ['old', 'new'] and H.mae.tolist() == [16.4, 15.7]
    assert H.verdict.iloc[1] == 'лучше' and pd.isna(H.mae_null.iloc[0])
