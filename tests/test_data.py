import os

import numpy as np
import pandas as pd
from PIL import Image

from anemia.data import build_manifest, check_no_leak


def _img(path, seed):
    rng = np.random.default_rng(seed)
    a = (rng.random((48, 64, 3)) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.fromarray(a).save(path)


def make_phone(root, n_patients=12, photos=2):
    rows = []
    for p in range(n_patients):
        for k in range(photos):
            f = f'img/p{p}_{k}.png'
            _img(os.path.join(root, 'phone', f), seed=p * 10 + k)
            rows.append({'file': f, 'patient_id': p, 'hb_g_l': 95 if p % 2 else 135, 'sex': 'F', 'age': 30})
    # копия снимка пациента 0 под чужим ID — должна попасть в ту же группу
    Image.open(os.path.join(root, 'phone', 'img/p0_0.png')).save(os.path.join(root, 'phone', 'img/copy.png'))
    rows.append({'file': 'img/copy.png', 'patient_id': 99, 'hb_g_l': 135, 'sex': 'F', 'age': 30})
    pd.DataFrame(rows).to_csv(os.path.join(root, 'phone', 'hb.csv'), index=False)


def test_manifest_groups_and_folds(tmp_path):
    make_phone(tmp_path)
    df = build_manifest(str(tmp_path), n_folds=3)
    assert len(df) == 25
    assert df.loc[df.patient_id == 'PH_99', 'group'].item() == df.loc[df.image.str.endswith('p0_0.png'), 'group'].item()
    assert df.groupby('group').fold.nunique().max() == 1         # пациент целиком в одном фолде
    assert set(df.fold) == {0, 1, 2}
    assert check_no_leak(df) == []
    assert df.anemic.sum() == 12 and set(df.grade) == {'norm', 'moderate'}


def test_missing_data_dir_raises(tmp_path):
    import pytest
    with pytest.raises(FileNotFoundError):
        build_manifest(str(tmp_path))
