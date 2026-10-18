"""Ядро HemaVision (anemia/core.py, anemia/predict.py) и анализ крови (anemia/labs.py).
Синтетика: проверяется код и отсутствие утечки, не точность."""
import io

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from anemia import core as CORE
from anemia import predict as PR
from anemia import segment_baseline as SB
from anemia.labs import hb_to_g_l, lab_hb_report
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from test_regions import synthetic_eye

from anemia import v12_model as V12


@pytest.fixture(autouse=True)
def small_models(monkeypatch):
    """Малые модели: проверяется устройство кода, а не качество, и тесты должны идти секунды."""
    monkeypatch.setattr(V12, '_clf', lambda: HistGradientBoostingClassifier(max_iter=10, random_state=0))
    monkeypatch.setattr(CORE, '_hb_model', lambda: HistGradientBoostingRegressor(max_iter=10, random_state=0))


def _synthetic(n=80, seed=0):
    rng = np.random.default_rng(seed)
    hb = rng.uniform(80, 160, n)
    rows, srows = [], []
    for i in range(n):
        y = 0 if hb[i] >= 120 else 1 if hb[i] >= 110 else 2
        meta = {'group': f'p{i}', 'fold': i % 5, 'y': y, 'anemic': y > 0, 'hb_g_l': hb[i], 'male': 0.0}
        for a in range(3):
            rows.append({**meta, 'aug': a, 'c_x': -hb[i] / 20 + rng.normal(0, 0.5), 'c_z': rng.normal(),
                         's_x': np.nan if i % 7 == 0 else rng.normal()})
        for c in range(2):
            srows.append({**meta, 'cond': f'c{c}', 'c_x': -hb[i] / 20 + rng.normal(0, 0.5), 'c_z': rng.normal(),
                          's_x': rng.normal()})
    return pd.DataFrame(rows), pd.DataFrame(srows)


def test_feature_list_by_dataset():
    cols = ['c_a', 's_a', 'w_a', 'male', 'group']
    assert CORE.feature_list(cols, 'eyes_defy') == ['c_a', 's_a', 'w_a', 'male']
    assert CORE.feature_list(cols, 'cp_anemic') == ['c_a']


def test_fit_core_thresholds_and_hb_rule():
    F, _ = _synthetic()
    B = CORE.fit_core(F, ['c_x', 'c_z', 's_x', 'male'], 'eyes_defy')
    assert B['t_lo'] <= B['t_hi'] and B['n_patients'] == 80
    assert B['hb_ok'] == (B['hb_skill_cv'] >= CORE.HB_MIN_SKILL)
    p, lv, hb, ood = CORE.apply_core(B, F[F.aug == 0][B['feats']].values)
    assert ((0 <= p) & (p <= 1)).all() and set(np.unique(lv)) <= {0, 1, 2} and len(hb) == 80
    assert np.isfinite(CORE.apply_core(B, [[np.nan, 0, np.nan, 0]])[0]).all()      # пропуски склеры — не ошибка


def test_cross_validate_never_trains_on_test_patients(monkeypatch):
    F, S = _synthetic()
    seen = []
    real = CORE.fit_core

    def spy(T, *a, **k):
        seen.append(set(T.group))
        return real(T, *a, **k)

    monkeypatch.setattr(CORE, 'fit_core', spy)
    O, SP = CORE.cross_validate(F, ['c_x', 'c_z', 's_x', 'male'], 'eyes_defy', S)
    for f, groups in zip(sorted(O.fold.unique()), seen):
        assert not groups & set(O.group[O.fold == f])
    r = CORE.core_metrics(O, SP)
    assert r['n'] == 80 and abs(r['share_low'] + r['share_mid'] + r['share_high'] - 1) < 1e-9
    assert 0 <= r['light_level_change'] <= 1


def test_ood_share_ignores_nan():
    lo, hi = np.zeros(3), np.ones(3)
    assert CORE.ood_share([0.5, np.nan, 5.0], lo, hi)[0] == pytest.approx(0.5)
    assert CORE.ood_share([np.nan] * 3, lo, hi)[0] == 1.0


# ------------------------------------------------------------------ чтение и QC фото
def test_load_photo_inputs_and_errors(tmp_path):
    img, _, _ = synthetic_eye()
    u8 = (img * 255).astype(np.uint8)
    p = tmp_path / 'eye.jpg'
    Image.fromarray(u8).save(p)
    assert PR.load_photo(str(p)).shape == u8.shape
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, 'PNG')
    assert PR.load_photo(buf.getvalue()).shape == u8.shape
    assert PR.load_photo(u8).dtype == np.float32
    rgba = np.dstack([u8, np.where(np.arange(u8.shape[1])[None] < 50, 0, 255).repeat(u8.shape[0], 0)]).astype(np.uint8)
    assert PR.load_photo(rgba)[:, :50].max() == 0                         # прозрачный фон → чёрный
    with pytest.raises(PR.PhotoError):
        PR.load_photo(str(tmp_path / 'нет.jpg'))
    (tmp_path / 'bad.jpg').write_bytes(b'not an image')
    with pytest.raises(PR.PhotoError):
        PR.load_photo(str(tmp_path / 'bad.jpg'))
    with pytest.raises(PR.PhotoError):
        PR.load_photo(u8[..., 0])


def test_frame_issues():
    img, _, _ = synthetic_eye()
    assert PR.frame_issues(img) == []
    assert PR.frame_issues(img[:20, :20])                                 # мало
    assert PR.frame_issues(np.repeat(img.mean(-1, keepdims=True), 3, -1))  # без цвета
    assert PR.frame_issues(img * 0.05)                                    # темно


def _bundle():
    F, _ = _synthetic()
    F = F.rename(columns={'c_x': 'c_EI_m', 'c_z': 'c_a_m', 's_x': 's_EI_m'})
    img, _, conj = synthetic_eye()
    return {'models': {'eyes_defy': CORE.fit_core(F, ['c_EI_m', 'c_a_m', 's_EI_m', 'male'], 'eyes_defy')},
            'conj_seg': SB.fit_samples([SB.image_samples(img, conj > 0, 0)])}


def test_analyze_photo_contract():
    B = _bundle()
    bad = PR.analyze_photo(b'garbage', bundle=B)
    assert bad['status'] == 'retake' and bad['risk'] is None and bad['hb_range'] is None and bad['issues']
    img, _, _ = synthetic_eye()
    r = PR.analyze_photo((img * 255).astype(np.uint8), sex='Ж', age=30, bundle=B)
    assert r['status'] in ('ok', 'retake', 'unreliable')
    if r['status'] != 'ok':                       # без уровня риска — никогда нет и цифры Hb
        assert r['risk'] is None and r['hb_range'] is None and r['p_anemia'] is None
    else:
        assert r['risk'] in CORE.RISK_LEVELS
    assert isinstance(PR.format_result(r), str)
    with pytest.raises(KeyError):
        PR.analyze_photo(img, age=2, bundle=B)    # модели детей в этом наборе нет


# ------------------------------------------------------------------ анализ крови
@pytest.mark.parametrize('value,unit,g_l', [(125, None, 125), ('12,5 г/дл', None, 125), (12.5, 'g/dL', 125),
                                            ('7.8', 'ммоль/л', 7.8 * 16.11), ('118 г/л', None, 118)])
def test_hb_units(value, unit, g_l):
    assert hb_to_g_l(value, unit) == pytest.approx(g_l)


@pytest.mark.parametrize('value,unit', [(12.5, None), (125, 'g/dL'), ('abc', None), (5, 'mg')])
def test_hb_units_rejected(value, unit):
    with pytest.raises(ValueError):
        hb_to_g_l(value, unit)


def test_lab_report_who():
    r = lab_hb_report(125, 'г/л', sex='M', age=40)
    assert r['grade'] == 'mild' and r['anemic'] and r['normal_from_g_l'] == 130
    assert lab_hb_report('10.5 g/dl', age=3)['grade'] == 'mild'
    assert not lab_hb_report(115, sex='F', age=30, pregnant=True)['anemic']


# ------------------------------------------------------------------ ткани по отдельности
def test_usable_requires_tissue_of_model():
    F, _ = _synthetic()
    assert CORE.usable(F, 'eyes_defy_sclera').s_x.notna().all()
    assert len(CORE.usable(F, 'eyes_defy_sclera')) < len(CORE.usable(F, 'eyes_defy_conj')) == len(F)
    assert CORE.feature_list(['c_a', 's_a', 'w_a'], 'eyes_defy_sclera') == ['s_a', 'male']
    assert CORE.feature_list(['c_a', 's_a', 'w_a'], 'eyes_defy_conj') == ['c_a', 'male']


def _fake_det(use, eye='весь кадр'):
    img, conj, sclera = synthetic_eye()
    crop = img.astype(np.float32)
    return {'crop': crop, 'conj': conj if 'conj' in use else np.zeros_like(conj),
            'sclera': sclera if 'sclera' in use else np.zeros_like(sclera), 'use': set(use), 'notes': [],
            'kind': 'глаз крупным планом', 'src_side': 1000, 'iris': None, 'eye': eye, 'score': len(use)}


def test_model_follows_found_tissues(monkeypatch):
    F, _ = _synthetic()
    F = F.rename(columns={'c_x': 'c_EI_m', 'c_z': 'c_a_m', 's_x': 's_EI_m'})
    full = {'models': {'eyes_defy': CORE.fit_core(F, ['c_EI_m', 'c_a_m', 's_EI_m', 'male'], 'eyes_defy')}}
    S = CORE.usable(F, 'eyes_defy_sclera')
    with_s = {'models': {**full['models'], 'eyes_defy_sclera': CORE.fit_core(S, ['s_EI_m', 'male'], 'eyes_defy_sclera')}}
    img, _, _ = synthetic_eye()
    monkeypatch.setattr(PR, 'quality', lambda *a: [])          # синтетика: проверяется маршрут, а не качество
    monkeypatch.setattr(PR, '_detect', lambda rgb, lm=None: _fake_det({'sclera'}))
    monkeypatch.setattr(PR, 'learned_conj', lambda det, seg: det)   # маршрут моделей; сегментатор — test_eye_location
    full['conj_seg'] = with_s['conj_seg'] = 'stub'
    r = PR.analyze_photo(img, sex='F', age=30, bundle=full, landmarker=None)
    assert r['stage'] == 'no_model' and r['risk'] is None          # модели «только склера» нет — не подменяем
    r = PR.analyze_photo(img, sex='F', age=30, bundle=with_s, landmarker=None)
    assert r['tissues'] == ['sclera'] and 'только склера' in r['model']
    assert r['features'] and not any(k.startswith('c_') for k in r['features'])
    monkeypatch.setattr(PR, '_detect', lambda rgb, lm=None: _fake_det(set(), eye=None))
    r = PR.analyze_photo(img, bundle=with_s, landmarker=None)
    assert r['stage'] == 'no_eye' and r['status'] == 'retake'
    monkeypatch.setattr(PR, '_detect', lambda rgb, lm=None: _fake_det(set()))
    assert PR.analyze_photo(img, bundle=with_s, landmarker=None)['stage'] == 'no_tissue'


def test_cross_validate_train_by_fold():
    """Признаки обучения, зависящие от тестового фолда (маски сегментатора без него): модель фолда f учится только
    на переданных строках; пациент теста среди них — ошибка."""
    F, _ = _synthetic()
    feats = ['c_x', 'c_z', 'male']
    train = {f: F[F.fold != f] for f in sorted(F.fold.unique())}
    O, _ = CORE.cross_validate(F, feats, 'eyes_defy', train_by_fold=train)
    O2, _ = CORE.cross_validate(F, feats, 'eyes_defy')
    assert np.allclose(O.p.values, O2.p.values)
    with pytest.raises(AssertionError):
        CORE.cross_validate(F, feats, 'eyes_defy', train_by_fold={f: F for f in train})
