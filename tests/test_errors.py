"""anemia.errors на синтетике: проверяется код, не точность."""
import numpy as np
import pandas as pd

from anemia import errors as E


def _oof(n=60, seed=0, noise=0.3):
    rng = np.random.default_rng(seed)
    y = np.arange(n) % 3
    P = np.full((n, 3), noise / 2)
    P[np.arange(n), y] = 1 - noise
    P += rng.uniform(0, 0.05, P.shape)
    P /= P.sum(1, keepdims=True)
    return pd.DataFrame({'group': [f'p{i}' for i in range(n)], 'y': y, 'anemic': y > 0,
                         'P': list(P), 'thr': np.full(n, 0.5)})


def test_confusion_counts():
    C = E.confusion([0, 0, 1, 2, 2], [0, 1, 1, 2, 0])
    assert C.values.sum() == 5 and np.trace(C.values) == 3
    assert C.iloc[2, 0] == 1


def test_summary_and_decide():
    O = _oof()
    r = E.summary(O, use_thr=False)
    assert r['bal_acc'] == 1.0 and r['sens'] == 1.0 and r['spec'] == 1.0
    assert (E.decide(O, True) > 0).tolist() == O.anemic.tolist()


def test_expected_error_ranks_confident_mistakes_first():
    O = _oof()
    O.at[0, 'P'] = np.array([0.0, 0.1, 0.9])    # норма, уверенно «умеренная+»
    e = E.expected_error(O)
    assert e.argmax() == 0 and abs(e[0] - 1.9) < 1e-9


def test_paired_delta_zero_on_identical():
    O = _oof()
    d = E.paired_delta(O, O.copy(), use_thr=False, n_boot=50)
    assert d['d_bal_acc'] == 0 and d['d_auc'] == 0


def test_breakdown_flags_small_groups():
    O = _oof()
    meta = pd.DataFrame({'group': O.group, 'пол': ['M'] * 50 + ['F'] * 10})
    B = E.breakdown({'врач': O}, meta, 'пол')
    assert set(B.группа) == {'M', 'F'}
    assert B.set_index('группа').loc['F', 'примечание'] == 'мало данных'


def test_panel_shape():
    crop = np.zeros((40, 30, 3), np.uint8)
    m = np.zeros((40, 30), bool)
    m[10:20, 5:15] = True
    img = E.panel(crop, [m, m], ['a', 'b', 'c'])
    assert img.shape == (62, 90, 3)
