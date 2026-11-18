"""Поправка ВОЗ 2024 на высоту проживания (таблица 4) в Я3: только к шкале ВОЗ и только если высота указана."""
import itertools

import pytest

from anemia.labs import (
    LabDataError,
    classify_kz2017,
    classify_who2024,
    lab_hb_report,
    lab_report,
    who2024_altitude_adjustment,
)


@pytest.mark.parametrize('h,adj', [(-100, 0), (0, 0), (499, 0), (499.9, 0), (500, 4), (550, 4), (999, 4),
                                   (1000, 8), (1499, 8), (1500, 11), (1999, 11), (2000, 14), (2499, 14),
                                   (2500, 18), (2999, 18), ('550,5', 4), (' 1200 ', 8)])
def test_altitude_table(h, adj):
    assert who2024_altitude_adjustment(h) == adj


@pytest.mark.parametrize('h', [None, '', '   ', float('nan')])
def test_no_altitude_means_no_adjustment(h):
    assert who2024_altitude_adjustment(h) is None
    c = classify_who2024(122, 'F', 30, False, None, h)
    assert c['altitude'] is None and c['normal_from_g_l'] == 120 and c['grade'] == 'norm'


GRID = list(itertools.product([69.5, 80, 99.5, 104.5, 109.5, 110, 119.5, 120, 129.5, 130, 140],
                              [('F', 30, False, None), ('M', 40, False, None), ('F', 28, True, 2),
                               ('F', 3, None, None), ('F', 8, None, None), ('M', 13, False, None)]))


@pytest.mark.parametrize('hb,who', GRID)
def test_without_altitude_results_are_unchanged(hb, who):
    """Без высоты — ровно прежний результат (как до появления поправки)."""
    assert classify_who2024(hb, *who) == classify_who2024(hb, *who, elevation_m=None)


@pytest.mark.parametrize('hb,who', GRID)
@pytest.mark.parametrize('h', [550, 1250, 2700])
def test_altitude_equals_shifted_thresholds(hb, who, h):
    """С высотой: категория как у Hb − поправка без высоты; порог нормы выше на поправку."""
    adj = who2024_altitude_adjustment(h)
    with_h, shifted = classify_who2024(hb, *who, elevation_m=h), classify_who2024(hb - adj, *who)
    if shifted['status'] == 'invalid':        # Hb − поправка вне правдоподобного диапазона — не наш случай
        pytest.skip('Hb − поправка < 30 г/л')
    assert (with_h['status'], with_h['grade']) == (shifted['status'], shifted['grade'])
    if with_h['status'] == 'ok':
        assert with_h['normal_from_g_l'] == shifted['normal_from_g_l'] + adj
        assert with_h['altitude'] == {'elevation_m': float(h), 'adjust_g_l': adj}


def test_karaganda_example():
    c = classify_who2024(122, 'F', 30, False, elevation_m=550)
    assert (c['grade'], c['normal_from_g_l']) == ('mild', 124)       # без высоты 122 — норма
    assert any('+4 г/л' in n and '550 м' in n and 'таблица 4' in n for n in c['notes'])
    assert classify_who2024(124, 'F', 30, False, elevation_m=550)['grade'] == 'norm'


def test_below_500_says_zero_adjustment():
    c = classify_who2024(122, 'F', 30, False, elevation_m=350)
    assert c['grade'] == 'norm' and c['altitude']['adjust_g_l'] == 0 and any('0 г/л' in n for n in c['notes'])


@pytest.mark.parametrize('h,status,words', [(3000, 'insufficient', '3000'), (5000, 'insufficient', 'не задана'),
                                            ('abc', 'invalid', 'высота'), (-600, 'invalid', 'высота'),
                                            (10000, 'invalid', 'высота'), (True, 'invalid', 'высота')])
def test_bad_or_unsupported_altitude_refused(h, status, words):
    c = classify_who2024(122, 'F', 30, False, elevation_m=h)
    assert c['status'] == status and words in c['message'] and c['grade'] is None
    with pytest.raises(LabDataError, match='высот'):
        lab_report(122, 'г/л', 'F', 30, False, elevation_m=h)


def test_kz_scale_and_who2011_not_adjusted():
    r = lab_report(118, 'г/л', 'F', 30, False, elevation_m=1200)
    assert r['kz2017'] == classify_kz2017(118, 'F', 30, False)                # шкала РК без поправки
    assert r['kz2017']['normal_from_g_l'] == 120 and r['who2024']['normal_from_g_l'] == 128
    assert 'применена только к ВОЗ 2024' in r['text'] and 'курение не применяется' in r['text']
    assert lab_hb_report(118, 'г/л', 'F', 30, False)['normal_from_g_l'] == 120   # ВОЗ 2011 (Я5) прежняя


def test_report_text_names_threshold_with_altitude():
    r = lab_report(12.6, 'г/дл', 'M', 40, False, elevation_m='550')
    who = next(x for x in r['text'].split('\n') if x.startswith('• ВОЗ 2024'))
    assert 'норма ≥ 134 г/л' in who and 'с учётом высоты проживания 550 м: +4 г/л' in who
