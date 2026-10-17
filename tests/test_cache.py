"""Кэш признаков и запись запусков. Синтетика: проверяется код (совпадение с кэшем и без), не точность."""
import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from anemia import runs as RUNS
from anemia.cache import FeatureCache, dataset_fingerprint
from anemia.data import build_manifest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))


# ------------------------------------------------------------------ FeatureCache
def _compute(x):
    _compute.calls += 1
    return {'v': x * 2}, {'read': 0.0, 'features': 0.0}


def _fetch(cache, img, params, arrays=None):
    return cache.fetch('t', {'image': str(img)}, arrays or {}, params, _compute, params['x'])


def test_cache_hit_and_invalidation(tmp_path):
    img = tmp_path / 'a.png'
    img.write_bytes(b'one')
    c = FeatureCache(str(tmp_path / 'cache'), 'on', salt={'script': 's1'})
    _compute.calls = 0
    v1, t1 = _fetch(c, img, {'x': 1})
    v2, t2 = _fetch(c, img, {'x': 1})
    assert v1 == v2 == {'v': 2} and (t1['miss'], t2['hit']) == (1, 1) and _compute.calls == 1
    _fetch(c, img, {'x': 2})                                       # другой параметр
    assert _compute.calls == 2
    _fetch(c, img, {'x': 1}, {'mask': np.ones((3, 3), bool)})      # другой массив (маска)
    assert _compute.calls == 3
    img.write_bytes(b'two')                                        # изменили исходный файл
    assert _fetch(c, img, {'x': 1})[1]['miss'] == 1 and _compute.calls == 4
    c2 = FeatureCache(c.root, 'on', salt={'script': 's2'})         # другая версия скрипта
    assert _fetch(c2, img, {'x': 1})[1]['miss'] == 1 and _compute.calls == 5
    assert _fetch(c, img, {'x': 1})[1]['hit'] == 1 and _compute.calls == 5


def test_cache_off_refresh_and_corrupt(tmp_path):
    img = tmp_path / 'a.png'
    img.write_bytes(b'one')
    root = str(tmp_path / 'cache')
    _compute.calls = 0
    _fetch(FeatureCache(root, 'off'), img, {'x': 1})
    assert not os.path.exists(root) and _compute.calls == 1        # off: ничего не пишет
    c = FeatureCache(root, 'on')
    _fetch(c, img, {'x': 1})
    assert _fetch(FeatureCache(root, 'refresh'), img, {'x': 1})[1]['miss'] == 1 and _compute.calls == 3
    key, _ = c.key('t', {'image': str(img)}, {}, {'x': 1})
    with open(c.path('t', key), 'wb') as f:
        f.write(b'not a pickle')
    v, t = _fetch(c, img, {'x': 1})                               # испорченный файл → пересчёт
    assert v == {'v': 2} and t['miss'] == 1 and _fetch(c, img, {'x': 1})[1]['hit'] == 1
    with pytest.raises(ValueError):
        FeatureCache(root, 'maybe')


def test_dataset_fingerprint(tmp_path):
    (tmp_path / 'a.png').write_bytes(b'one')
    df = pd.DataFrame({'image': ['a.png'], 'fold': [0]})
    f0 = dataset_fingerprint(str(tmp_path), df, ['image'])
    assert f0 == dataset_fingerprint(str(tmp_path), df, ['image'])
    assert f0 != dataset_fingerprint(str(tmp_path), df.assign(fold=1), ['image'])
    (tmp_path / 'a.png').write_bytes(b'two')
    assert f0 != dataset_fingerprint(str(tmp_path), df, ['image'])


# ------------------------------------------------------------------ история и папки запусков
def test_append_history_backup_and_unique_runs(tmp_path):
    hist, bdir = tmp_path / 'h.csv', tmp_path / 'backups'
    pd.DataFrame({'timestamp': ['t0'], 'tag': ['old'], 'auc': [0.7]}).to_csv(hist, index=False)
    before = hist.read_bytes()
    cols = ['timestamp', 'tag', 'auc', 'mae']
    RUNS.append_history(str(hist), [{'auc': 0.8, 'mae': 12.0}], cols, {'timestamp': 't1', 'tag': 'new'}, str(bdir))
    h = pd.read_csv(hist)
    assert list(h.columns) == cols and h.tag.tolist() == ['old', 'new'] and h.auc.tolist() == [0.7, 0.8]
    backups = list(bdir.iterdir())
    assert len(backups) == 1 and backups[0].read_bytes() == before
    ids = {RUNS.create_run_dir(str(tmp_path / 'runs'), 'same tag')[0] for _ in range(5)}
    assert len(ids) == 5


def test_preserve_before_overwrite(tmp_path):
    runs, bdir = tmp_path / 'runs', tmp_path / 'backups'
    latest = tmp_path / 'oof_x.csv'
    assert RUNS.preserve_before_overwrite(str(latest), str(runs), str(bdir)) is None
    latest.write_text('old run without a run folder')
    assert open(RUNS.preserve_before_overwrite(str(latest), str(runs), str(bdir))).read() == latest.read_text()
    (runs / 'r1').mkdir(parents=True)
    (runs / 'r1' / 'oof_x.csv').write_text('copy')
    latest.write_text('copy')
    assert RUNS.preserve_before_overwrite(str(latest), str(runs), str(bdir)) is None


# ------------------------------------------------------------------ evaluate.py: с кэшем = без кэша
def _eyes_defy_like(root, n=8):
    """Раскладка Eyes-defy (фото + маска врача), чтобы пройти сегментацию, маски doctor/pred и v12."""
    from fetch_data import eyes_defy_labels
    raw = root / 'eyes_defy_anemia' / 'raw'
    rng = np.random.default_rng(0)
    for country in ('Italy', 'India'):
        recs = []
        for p in range(1, n + 1):
            d = raw / 'dataset' / country / str(p)
            os.makedirs(d, exist_ok=True)
            img = np.full((120, 160, 3), 200, np.uint8)
            img[40:70, 50:110] = (40, 30, 30)
            img[80:105, 30 + p:130 - p] = (190, 70 + 4 * p, 70)
            img = np.clip(img + rng.normal(0, 4, img.shape), 0, 255).astype(np.uint8)
            Image.fromarray(img).save(d / f'{p}.jpg')
            m = np.zeros_like(img)
            m[80:105, 30 + p:130 - p] = img[80:105, 30 + p:130 - p]
            Image.fromarray(m).save(d / f'{p}_palpebral.png')
            recs.append({'Number': f'{p:03d}', 'Hgb': str(9 + p * 0.6).replace('.', ','), 'Sex': 'F' if p % 2 else 'M'})
        pd.DataFrame(recs).to_excel(raw / 'dataset' / country / f'{country}.xlsx', index=False)
    eyes_defy_labels(str(raw)).to_csv(root / 'eyes_defy_anemia' / 'labels.csv', index=False)
    build_manifest(str(root), n_folds=3).to_csv(root / 'manifest.csv', index=False)


def _run(data, hist, *extra):
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'evaluate.py'), '--data', str(data),
                        '--model', 'ridge', '--tag', 'pytest cache', '--jobs', '1', '--history', str(hist), *extra],
                       capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 0, r.stderr[-3000:]
    run_dir = next(line.split(': ', 1)[1] for line in r.stdout.splitlines() if line.startswith('Результаты запуска: '))
    return run_dir, json.load(open(os.path.join(run_dir, 'run.json')))


def _hits(meta):
    return {t['stage']: (int(t['hit']), t['n']) for t in meta['timings']}


def test_evaluate_same_with_and_without_cache(tmp_path):
    data, hist = tmp_path / 'data', tmp_path / 'repo_history.csv'
    _eyes_defy_like(data)
    runs = [_run(data, hist, '--no-cache')]
    assert not os.path.exists(runs[0][1]['cache']['dir'])         # --no-cache ничего не пишет
    runs += [_run(data, hist), _run(data, hist)]
    cold, warm = runs[1][1], runs[2][1]
    # тёплый запуск: признаки Hb и v12 — из кэша; попиксельная сегментация не кэшируется
    for stage, (hit, n) in _hits(warm).items():
        assert hit == (0 if stage.startswith('сегментация') else n), stage
    assert sum(h for h, _ in _hits(cold).values()) == 0
    # метрики и OOF совпадают побайтно: с кэшем, без кэша, холодный и тёплый
    tabs = [pd.read_csv(os.path.join(d, 'metrics.csv')).drop(columns='run_id') for d, _ in runs]
    for t in tabs[1:]:
        pd.testing.assert_frame_equal(tabs[0], t, check_exact=True)
    oofs = sorted(f for f in os.listdir(runs[0][0]) if f.startswith('oof_'))
    assert oofs
    for f in oofs:
        ref = open(os.path.join(runs[0][0], f), 'rb').read()
        assert all(open(os.path.join(d, f), 'rb').read() == ref for d, _ in runs[1:]), f
    # три запуска — три папки и три набора строк в обеих историях, ничего не перезаписано
    ids = [m['run_id'] for _, m in runs]
    assert len(set(ids)) == 3 and all(m['status'] == 'done' and m['commit'] for _, m in runs)
    dh = pd.read_csv(data / 'results' / 'metrics_history.csv')
    assert dh.run_id.drop_duplicates().tolist() == ids and len(dh) == 3 * len(tabs[0])
    assert len(pd.read_csv(hist)) == 3 * len(tabs[0])
    # изменили один снимок → только он пересчитывается
    lab = pd.read_csv(data / 'eyes_defy_anemia' / 'labels.csv')
    p = data / 'eyes_defy_anemia' / 'raw' / lab.image[0]
    Image.fromarray(255 - np.asarray(Image.open(p))).save(p)
    _, changed = _run(data, hist)
    for stage, (hit, n) in _hits(changed).items():
        if stage.startswith('сегментация'):
            assert hit == 0
        elif 'маска pred' in stage:              # сегментатор переобучен с новым снимком: маски могли измениться
            assert hit <= n - 1, stage
        else:
            assert hit == n - 1, stage
    _, refreshed = _run(data, hist, '--refresh-cache')
    assert sum(h for h, _ in _hits(refreshed).values()) == 0


def test_error_analysis_cache_freshness(tmp_path):
    from error_analysis import ea_cache_fresh, ea_cache_path, ea_fingerprint, ea_rows
    (tmp_path / 'a.jpg').write_bytes(b'photo')
    (tmp_path / 'a_m.png').write_bytes(b'mask')
    man = pd.DataFrame({'dataset': ['eyes_defy'], 'image': ['a.jpg'], 'mask_conj': ['a_m.png'], 'fold': [0]})
    df = ea_rows(man)
    cache = ea_cache_path(str(tmp_path), 't')
    os.makedirs(os.path.dirname(cache))
    pd.to_pickle([], cache)
    assert not ea_cache_fresh(str(tmp_path), 't', df)              # старый кэш без ключа — не доверяем
    with open(cache + '.key', 'w') as f:
        f.write(ea_fingerprint(str(tmp_path), df))
    assert ea_cache_fresh(str(tmp_path), 't', df)
    assert not ea_cache_fresh(str(tmp_path), 't', df.assign(fold=1))
    (tmp_path / 'a_m.png').write_bytes(b'other mask')
    assert not ea_cache_fresh(str(tmp_path), 't', df)
