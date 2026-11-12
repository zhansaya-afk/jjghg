"""scripts/subgroup_report.py: таблица по странам/полу из OOF evaluate --task core. Таблицы — игрушечные, проверяется
арифметика счёта (FP среди здоровых, отказы, сдвиг Hb), не точность модели."""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'scripts'))
import subgroup_report as SR  # noqa: E402


def _toy():
    man = pd.DataFrame({'dataset': 'eyes_defy',
                        'patient_id': ['ED_India_1', 'ED_India_2', 'ED_India_3', 'ED_Italy_1', 'ED_Italy_2', 'ED_Italy_3'],
                        'sex': ['F', 'F', 'M', 'M', 'M', 'F']})
    man['group'] = man.patient_id
    oof = pd.DataFrame({'group': ['ED_India_1', 'ED_India_2', 'ED_Italy_1', 'ED_Italy_2', 'ED_Italy_3'],
                        'anemic': [False, True, False, False, True],
                        'level': [2, 1, 0, 1, 0],
                        'p': [0.9, 0.6, 0.1, 0.5, 0.2],
                        'hb_g_l': [130.0, 100.0, 150.0, 140.0, 110.0],
                        'hb_pred': [110.0, 110.0, 140.0, 140.0, 120.0],
                        'accepted': [True, True, True, True, True]})
    return man, oof


def test_counts_by_country():
    man, oof = _toy()
    t = SR.subgroup_table(oof, man).set_index('группа')
    india, italy = t.loc['страна: India'], t.loc['страна: Italy']
    assert (india.n_всего, india.оценено, india.отказ) == (3, 2, 1)       # ED_India_3 — нет в OOF → отказ
    assert (india.здоровых, india.FP_средний_плюс, india.FP_высокий) == (1, 1, 1)
    assert (italy.здоровых, italy.FP_средний_плюс, italy.FP_высокий, italy.FN_низкий) == (2, 1, 0, 1)
    assert italy.спец == pytest.approx(0.5) and italy.чувств == pytest.approx(0.0)
    assert india.сдвиг_Hb_здоровых == pytest.approx(-20.0)               # 110 − 130
    assert italy.сдвиг_Hb == pytest.approx((-10 + 0 + 10) / 3)


def test_rows_for_sex_and_all():
    man, oof = _toy()
    t = SR.subgroup_table(oof, man).set_index('группа')
    assert t.loc['все', 'n_всего'] == 6 and t.loc['все', 'оценено'] == 5
    assert t.loc['пол: F', 'здоровых'] == 1 and t.loc['пол: M', 'здоровых'] == 2
    assert 'страна × пол: India F' in t.index


def test_ood_rows_count_as_refused_and_metrics_nan_when_no_class():
    man, oof = _toy()
    oof.loc[oof.group == 'ED_Italy_2', 'accepted'] = False                 # OOD → отказ, в метрики не идёт
    t = SR.subgroup_table(oof, man).set_index('группа')
    assert t.loc['страна: Italy', 'отказ'] == 1 and t.loc['страна: Italy', 'здоровых'] == 1
    assert np.isnan(t.loc['страна × пол: India M', 'спец'])                 # там никого не оценили — «—», не 0


def test_patient_in_two_models_is_an_error():
    man, oof = _toy()
    with pytest.raises(AssertionError, match='двумя моделями'):
        SR.subgroup_table(pd.concat([oof, oof.iloc[:1]]), man)
