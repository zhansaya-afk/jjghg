"""v12 в evaluate.py: фолды по пациенту, порог только из обучающих фолдов. Синтетика: проверяется код, не точность."""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from evaluate import v12_cv, v12_decide, v12_metrics  # noqa: E402

from anemia import v12_model as V12  # noqa: E402


def _synthetic(n=60, seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n)
    rows, srows = [], []
    for i in range(n):
        meta = {'group': f'p{i}', 'fold': i % 5, 'y': int(y[i]), 'anemic': bool(y[i] > 0), 'male': 0.5}
        for a in range(3):
            rows.append({**meta, 'aug': a, 'c_x': y[i] + rng.normal(0, 0.7), 'c_z': rng.normal()})
        for c in range(2):
            srows.append({**meta, 'cond': f'c{c}', 'c_x': y[i] + rng.normal(0, 0.7), 'c_z': rng.normal()})
    return pd.DataFrame(rows), pd.DataFrame(srows)


def test_v12_cv_patient_folds_and_metrics():
    F, S = _synthetic()
    O, SP = v12_cv(F, S, ['c_x', 'c_z'], 0.80)
    assert len(O) == 60 and O.group.is_unique
    assert O.groupby('fold').thr.nunique().eq(1).all()          # один порог на тестовый фолд
    for use_thr in (False, True):
        r = v12_metrics(O, SP, use_thr, 0.80)
        assert 0 <= r['bal_acc'] <= 1 and 0 <= r['light_flip'] <= 1 and r['n'] == 60


def test_v12_decide_rules():
    P = np.array([[0.7, 0.2, 0.1], [0.4, 0.1, 0.5], [0.9, 0.05, 0.05]])
    assert v12_decide(P).tolist() == [0, 2, 0]
    assert v12_decide(P, 0.25).tolist() == [1, 2, 0]             # выше порога, но argmax «норма» → «лёгкая»


def test_who_classes():
    assert V12.hb_class(12.5, *V12.who_thresholds('M', 30)) == 1
    assert V12.hb_class(12.5, *V12.who_thresholds('F', 30)) == 0
    assert V12.hb_class(9.5, *V12.who_thresholds(None, 2)) == 2
