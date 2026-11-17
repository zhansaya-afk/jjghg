"""Анализ крови (Я3): ВОЗ 2024 и протокол МЗ РК 2017 — две отдельные системы; неизвестное не подставляется."""
import pytest

from anemia import labs as L
from anemia.labs import LabDataError, classify_kz2017, classify_who2024, lab_report


# ------------------------------------------------------------------ ВОЗ 2024: группы и границы (таблица 3)
@pytest.mark.parametrize('hb,sex,age,preg,tri,grade', [
    # мужчины 15–65: ≥130 / 110–129 / 80–109 / <80
    (130, 'M', 40, None, None, 'norm'), (129.9, 'M', 40, None, None, 'mild'), (110, 'M', 40, None, None, 'mild'),
    (109.9, 'M', 40, None, None, 'moderate'), (80, 'M', 40, None, None, 'moderate'),
    (79.9, 'M', 40, None, None, 'severe'),
    # женщины 15–65, не беременные: ≥120 / 110–119 / 80–109 / <80
    (120, 'F', 40, False, None, 'norm'), (119.9, 'F', 40, False, None, 'mild'), (110, 'F', 40, False, None, 'mild'),
    (109.9, 'F', 40, False, None, 'moderate'), (80, 'F', 40, False, None, 'moderate'),
    (79.9, 'F', 40, False, None, 'severe'),
    # беременные: I и III ≥110 / 100–109 / 70–99 / <70; II ≥105 / 95–104 / 70–94 / <70
    (110, 'F', 28, True, 1, 'norm'), (109.9, 'F', 28, True, 1, 'mild'), (100, 'F', 28, True, 1, 'mild'),
    (99.9, 'F', 28, True, 1, 'moderate'), (70, 'F', 28, True, 1, 'moderate'), (69.9, 'F', 28, True, 1, 'severe'),
    (105, 'F', 28, True, 2, 'norm'), (104.9, 'F', 28, True, 2, 'mild'), (95, 'F', 28, True, 2, 'mild'),
    (94.9, 'F', 28, True, 2, 'moderate'), (70, 'F', 28, True, 2, 'moderate'), (69.9, 'F', 28, True, 2, 'severe'),
    (110, 'F', 28, True, 3, 'norm'), (109.9, 'F', 28, True, 3, 'mild'), (100, 'F', 28, True, 3, 'mild'),
    (99.9, 'F', 28, True, 3, 'moderate'), (69.9, 'F', 28, True, 3, 'severe'),
    # дети (пол на порог не влияет): 6–23 мес ≥105 / 95–104 / 70–94; 24–59 мес ≥110 / 100–109 / 70–99
    (105, None, 0.5, None, None, 'norm'), (104.9, None, 0.5, None, None, 'mild'),
    (94.9, None, 1.99, None, None, 'moderate'), (69.9, None, 1, None, None, 'severe'),
    (109.9, None, 2, None, None, 'mild'), (99.9, None, 4.99, None, None, 'moderate'),
    (69.9, None, 3, None, None, 'severe'),
    # 5–11 лет ≥115 / 110–114 / 80–109 / <80; 12–14 лет ≥120 / 110–119 / 80–109 / <80
    (115, None, 5, None, None, 'norm'), (114.9, 'F', 11.99, None, None, 'mild'), (79.9, 'M', 8, None, None, 'severe'),
    (120, 'M', 12, None, None, 'norm'), (119.9, 'F', 14.99, False, None, 'mild'),
    (109.9, 'M', 13, None, None, 'moderate'),
    # 15 лет — уже взрослые нормы
    (125, 'M', 15, None, None, 'mild'), (125, 'M', 14.99, None, None, 'norm'),
])
def test_who2024_grades_and_boundaries(hb, sex, age, preg, tri, grade):
    c = classify_who2024(hb, sex, age, preg, tri)
    assert c['status'] == 'ok', c['message']
    assert c['grade'] == grade and c['anemic'] is (grade != 'norm')


@pytest.mark.parametrize('sex,age,preg,tri,normal,group', [
    ('M', 40, None, None, 130, 'мужчины 15–65 лет'), ('F', 40, False, None, 120, 'женщины 15–65 лет, не беременные'),
    ('F', 28, True, 1, 110, 'беременные, I триместр'), ('F', 28, True, 2, 105, 'беременные, II триместр'),
    ('F', 28, True, 3, 110, 'беременные, III триместр'), (None, 1, None, None, 105, 'дети 6–23 мес'),
    (None, 3, None, None, 110, 'дети 24–59 мес'), (None, 7, None, None, 115, 'дети 5–11 лет'),
    ('M', 13, None, None, 120, 'дети 12–14 лет')])
def test_who2024_normal_threshold_by_group(sex, age, preg, tri, normal, group):
    c = classify_who2024(140, sex, age, preg, tri)
    assert c['normal_from_g_l'] == normal and c['group'] == group


def test_second_trimester_differs_from_first_and_third():
    assert classify_who2024(107, 'F', 28, True, 2)['grade'] == 'norm'
    assert classify_who2024(107, 'F', 28, True, 1)['grade'] == 'mild'
    assert classify_who2024(107, 'F', 28, True, 3)['grade'] == 'mild'
    assert classify_who2024(97, 'F', 28, True, 2)['grade'] == 'mild'
    assert classify_who2024(97, 'F', 28, True, 1)['grade'] == 'moderate'


# ------------------------------------------------------------------ неизвестное: не подставляется
def test_unknown_trimester_only_when_it_changes_the_category():
    c = classify_who2024(107, 'F', 28, True, None)
    assert c['status'] == 'insufficient' and 'триместр' in c['message'] and c['grade'] is None
    assert classify_who2024(97, 'F', 28, True, None)['status'] == 'insufficient'     # I/III умеренная, II лёгкая
    c = classify_who2024(115, 'F', 28, True, None)          # норма при любом триместре
    assert c['status'] == 'ok' and c['grade'] == 'norm' and c['normal_from_g_l'] == 110
    assert c['notes'] and 'одинакова' in c['notes'][0]
    assert classify_who2024(80, 'F', 28, True, None)['grade'] == 'moderate'


def test_unknown_sex_adult():
    c = classify_who2024(125, None, 30, False)
    assert c['status'] == 'insufficient' and 'пол' in c['message']
    assert classify_who2024(135, None, 30, False)['grade'] == 'norm'
    assert classify_who2024(100, None, 30, False)['grade'] == 'moderate'
    assert classify_who2024(100, None, 30, None)['status'] == 'insufficient'   # и беременность неизвестна


def test_unknown_pregnancy_in_women():
    c = classify_who2024(115, 'F', 30, None)
    assert c['status'] == 'insufficient' and 'беременност' in c['message']
    assert classify_who2024(140, 'F', 30, None)['grade'] == 'norm'
    assert classify_who2024(112, 'F', 8, None)['grade'] == 'mild'   # до 12 лет беременность не перебирается


@pytest.mark.parametrize('age,words', [(None, 'возраст'), ('', 'возраст'), (float('nan'), 'возраст'),
                                       (0.3, '6 месяцев'), (0, '6 месяцев')])
def test_unknown_or_too_young_age(age, words):
    c = classify_who2024(110, 'F', age, False)
    assert c['status'] == 'insufficient' and words in c['message'] and c['grade'] is None


# kz — ожидаемый статус шкалы РК: триместр она не принимает, поэтому противоречия триместра её не касаются
@pytest.mark.parametrize('kw,words,kz', [
    ({'sex': 'M', 'pregnant': True}, 'мужчин', 'invalid'),
    ({'sex': 'F', 'pregnant': False, 'trimester': 2}, 'триместр', 'ok'),
    ({'sex': 'M', 'trimester': 1}, 'триместр', 'ok'),
    ({'sex': 'F', 'pregnant': True, 'trimester': 4}, 'триместр', 'not_applicable'),
    ({'sex': 'F', 'pregnant': True, 'trimester': 'II'}, 'триместр', 'not_applicable'),
    ({'sex': 'X'}, 'пол', 'invalid'),
    ({'sex': 'F', 'age': -1}, 'возраст', 'invalid'),
    ({'sex': 'F', 'age': 121}, 'возраст', 'invalid'),
    ({'sex': 'F', 'age': 'abc'}, 'возраст', 'invalid'),
    ({'sex': 'F', 'age': True}, 'возраст', 'invalid'),
    ({'sex': 'F', 'age': 8, 'pregnant': True, 'trimester': 2}, 'возраст', 'invalid'),
])
def test_contradictory_or_invalid_inputs(kw, words, kz):
    args = {'age': 30, **kw}
    c = classify_who2024(110, **args)
    assert c['status'] == 'invalid' and words in c['message'] and c['grade'] is None
    k = classify_kz2017(110, args.get('sex'), args['age'], args.get('pregnant'))
    assert k['status'] == kz
    if kz == 'invalid':
        assert k['grade'] is None and k['anemic'] is None


@pytest.mark.parametrize('hb', [float('nan'), float('inf'), None, '', 'abc', True, 10, 29.9, 250.1, 400])
def test_classifiers_reject_bad_hb_instead_of_grading(hb):
    for c in (classify_who2024(hb, 'M', 40, False), classify_kz2017(hb, 'M', 40, False)):
        assert c['status'] == 'invalid' and 'Hb' in c['message']
        assert c['grade'] is None and c['anemic'] is None


def test_pregnancy_age_check_boundary():
    assert classify_who2024(110, 'F', 9.9, True, 1)['status'] == 'invalid'
    assert classify_who2024(110, 'F', 10, True, 1)['status'] == 'ok'


def test_kz_unknown_sex_shows_higher_threshold():
    c = classify_kz2017(100, None, 40, False)
    assert c['grade'] == 'mild' and c['normal_from_g_l'] == 130   # как у ВОЗ: при неизвестном поле — порог выше


def test_trimester_accepts_digit_strings_and_cyrillic_sex():
    assert classify_who2024(107, 'Ж', 28, 'да', '2')['grade'] == 'norm'
    assert classify_who2024(125, 'М', 40)['grade'] == 'mild'


def test_over_65_is_classified_with_a_note():
    c = classify_who2024(125, 'M', 70)
    assert c['grade'] == 'mild' and any('65' in n for n in c['notes'])


# ------------------------------------------------------------------ протокол МЗ РК 2017
@pytest.mark.parametrize('hb,sex,grade,anemic', [
    (120, 'F', 'norm', False), (119.9, 'F', 'mild', True), (90, 'F', 'mild', True), (89.9, 'F', 'moderate', True),
    (70, 'F', 'moderate', True), (69.9, 'F', 'severe', True),
    (130, 'M', 'norm', False), (120, 'M', 'mild', True), (89.9, 'M', 'moderate', True), (69.9, 'M', 'severe', True),
])
def test_kz2017_grades(hb, sex, grade, anemic):
    c = classify_kz2017(hb, sex, 40, False)
    assert c['status'] == 'ok' and c['grade'] == grade and c['anemic'] is anemic


def test_kz2017_men_120_129_anemic_but_ungraded():
    c = classify_kz2017(125, 'M', 40, None)
    assert c['status'] == 'ok' and c['anemic'] is True and c['grade'] is None and 'не задана' in c['grade_ru']


@pytest.mark.parametrize('sex,age,preg,status', [
    ('F', 17, False, 'not_applicable'), ('F', 30, True, 'not_applicable'), ('F', 30, None, 'insufficient'),
    (None, 30, None, 'insufficient'), ('F', None, False, 'insufficient'), ('M', 30, True, 'invalid')])
def test_kz2017_applicability(sex, age, preg, status):
    assert classify_kz2017(100, sex, age, preg)['status'] == status


def test_kz2017_unknown_sex():
    assert classify_kz2017(125, None, 40, False)['status'] == 'insufficient'
    assert classify_kz2017(115, None, 40, False)['grade'] == 'mild'
    assert classify_kz2017(135, None, 40, False)['grade'] == 'norm'


# ------------------------------------------------------------------ отчёт: две системы раздельно
def test_report_shows_who_and_kz_separately():
    r = lab_report(95, 'г/л', sex='F', age=30, pregnant=False)
    assert r['who2024']['grade'] == 'moderate' and r['kz2017']['grade'] == 'mild'     # категории разные
    assert r['who2024']['system'] == 'who2024' and r['kz2017']['system'] == 'kz2017'
    assert 'group' not in r['kz2017']
    lines = r['text'].split('\n')
    who = [x for x in lines if x.startswith('• ВОЗ 2024')]
    kz = [x for x in lines if x.startswith('• МЗ РК 2017')]
    assert len(who) == 1 and len(kz) == 1
    assert 'умеренная анемия' in who[0] and 'лёгкая' not in who[0]
    assert 'анемия лёгкой степени' in kz[0] and 'умеренная' not in kz[0]
    assert 'не применены' in r['text'] and 'не диагноз' in r['text']


def test_report_keeps_legacy_keys_from_who2024():
    r = lab_report(125, 'г/л', sex='M', age=40)
    for k in ('source', 'hb_g_l', 'grade', 'grade_ru', 'anemic', 'normal_from_g_l', 'group', 'text'):
        assert k in r
    assert (r['grade'], r['anemic'], r['normal_from_g_l'], r['status']) == ('mild', True, 130, 'ok')


def test_report_second_trimester():
    r = lab_report(107, 'г/л', sex='F', age=28, pregnant=True, trimester=2)
    assert r['grade'] == 'norm' and r['normal_from_g_l'] == 105 and 'II триместр' in r['text']
    assert r['kz2017']['status'] == 'not_applicable'


@pytest.mark.parametrize('kw,words', [
    ({'sex': None, 'age': 30, 'pregnant': False}, 'пол'),
    ({'sex': 'F', 'age': 28, 'pregnant': True}, 'триместр'),
    ({'sex': 'F', 'age': None, 'pregnant': False}, 'возраст'),
    ({'sex': 'M', 'age': 30, 'pregnant': True}, 'мужчин'),
])
def test_report_raises_clear_error_instead_of_guessing(kw, words):
    with pytest.raises(LabDataError, match=words) as e:
        lab_report(107 if kw.get('pregnant') else 125, 'г/л', **kw)
    assert isinstance(e.value, ValueError)                  # интерфейс ловит ValueError, как раньше
    assert '• ВОЗ 2024' in e.value.report['text'] and '• МЗ РК 2017' in e.value.report['text']
    assert e.value.report['grade'] is None


def test_report_shows_exact_value_not_rounded_up():
    r = lab_report(119.95, 'г/л', sex='F', age=30, pregnant=False)
    assert r['text'].startswith('Hb 119,95 г/л') and r['grade'] == 'mild'


# ------------------------------------------------------------------ единицы и границы Hb
@pytest.mark.parametrize('value,unit,want', [
    ('12,5 г/дл', None, 125), ('12.5', 'G/DL', 125), ('7,8', 'ммоль/л', 7.8 * 16.11), (125, 'г/л', 125),
    ('125', None, 125), (30, 'г/л', 30), (250, 'г/л', 250), ('12.5 g%', None, 125)])
def test_units(value, unit, want):
    assert L.hb_to_g_l(value, unit) == pytest.approx(want)


@pytest.mark.parametrize('value,unit', [('12', None), (7.8, None), (29.9, 'г/л'), (250.1, 'г/л'), (125, 'мг/дл'),
                                        ('abc', None), ('-5', None), ('', None), (float('nan'), 'г/л')])
def test_bad_units_and_values_rejected(value, unit):
    with pytest.raises(ValueError):
        lab_report(value, unit, sex='F', age=30, pregnant=False)


# ------------------------------------------------------------------ старые нормы (фото, метки, персонализатор) не тронуты
def test_legacy_who2011_functions_unchanged():
    assert L.who_limits('F', 30, True).normal == 110          # без триместров, как раньше
    assert L.who_limits('F', 30, None).normal == 120
    assert L.who_limits(None, 30).normal == 120
    assert L.who_limits(None, 1).normal == 110                # ВОЗ 2011: до 5 лет — 110
    assert L.who_grade(107, 'F', 28, True) == 'mild'


# ------------------------------------------------------------------ степень анемии РК ≠ диагноз ЖДА
@pytest.mark.parametrize('hb,sex,preg', [(65, 'F', False), (85, 'M', False), (105, 'F', False), (125, 'M', False),
                                         (140, 'M', False), (107, 'F', True)])
def test_iron_deficiency_is_separate_and_never_confirmed(hb, sex, preg):
    r = lab_report(hb, 'г/л', sex=sex, age=40, pregnant=preg, trimester=2 if preg else None)
    iron = r['iron_deficiency']
    assert iron == {'status': 'not_confirmed', 'confirmed': False,
                    'text': 'Железодефицит не подтверждён: недостаточно лабораторных данных '
                            '(нет ферритина и насыщения трансферрина, TSat).'}
    assert 'iron' not in r['kz2017'] and r['kz2017']['notes'] == []      # степень анемии — без статуса железа
    lines = r['text'].split('\n')
    kz = next(x for x in lines if x.startswith('• МЗ РК 2017'))
    assert 'степень анемии' in kz and 'ЖДА' not in kz                    # строка РК не называет ЖДА
    assert '• Железодефицит не подтверждён: недостаточно лабораторных данных' in r['text']
    assert lines.index(kz) < lines.index('• ' + iron['text'])             # отдельная строка после степени
    assert 'Степень анемии — не диагноз ЖДА' in r['text']


def test_iron_status_also_in_refused_report():
    with pytest.raises(LabDataError) as e:
        lab_report(107, 'г/л', sex='F', age=28, pregnant=True)
    assert e.value.report['iron_deficiency']['confirmed'] is False
