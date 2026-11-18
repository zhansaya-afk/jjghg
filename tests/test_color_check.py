"""Сверка формул цвета с независимым эталоном (anemia/color_check.py) и скрипт scripts/verify_color.py."""
import numpy as np

from anemia import color as C
from anemia import color_check as K


def test_reference_matches_literature():
    a = np.array([v[0] for v in K.LAB_ANCHORS.values()], float)
    lit = np.array([v[1] for v in K.LAB_ANCHORS.values()])
    assert K.delta_e76(K.ref_lab(a), lit).max() < 1e-3


def test_all_checks_pass():
    rows = K.verify(n_random=3000)
    bad = [r['проверка'] for r in rows if not r['прошла']]
    assert not bad, bad


def test_hue_of_primaries_exact():
    for rgb, h in K.HUE_ANCHORS.values():
        H, _, _ = C.rgb_to_hsi(np.array([[rgb]], np.float32))
        assert K.hue_diff(H[0, 0], h) < 1e-3


def test_sclera_balance_cancels_moderate_light_not_clipped_warm():
    T = {(r['T'], r['экспозиция'], r['способ'], r['ткань']): r for r in K.stability()}
    tis = 'конъюнктива, бледная'
    # 4500–7500 K без клиппинга: баланс по склере возвращает EI и оттенок почти точно
    for t in (4500, 5500, 7500):
        assert abs(T[(t, 1.0, 'баланс по склере', tis)]['EI'] - T[(6500, 1.0, 'баланс по склере', tis)]['EI']) < 0.01
    raw = abs(T[(4500, 1.0, 'сырой', tis)]['EI'] - T[(6500, 1.0, 'сырой', tis)]['EI'])
    assert raw > 0.03                                  # без баланса свет заметно сдвигает EI


def test_script_writes_outputs(tmp_path):
    import pytest
    pytest.importorskip("matplotlib")
    import importlib.util
    import pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / 'scripts' / 'verify_color.py'
    spec = importlib.util.spec_from_file_location('verify_color', p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    assert m.main(['--out', str(tmp_path)]) == 0
    for f in ('verify.csv', 'stability.csv', 'spread.csv', 'stability.png'):
        assert (tmp_path / f).stat().st_size > 0
