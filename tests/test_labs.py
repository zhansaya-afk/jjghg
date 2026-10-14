import pytest

from anemia.labs import is_anemic, who_grade, who_limits


@pytest.mark.parametrize('sex,age,preg,normal', [
    ('M', 30, False, 130), ('F', 30, False, 120), ('F', 28, True, 110),
    (None, 2, False, 110), (None, 8, False, 115), ('M', 13, False, 120)])
def test_who_normal_threshold(sex, age, preg, normal):
    assert who_limits(sex, age, preg).normal == normal


@pytest.mark.parametrize('hb,sex,age,preg,grade', [
    (135, 'M', 40, False, 'norm'), (125, 'M', 40, False, 'mild'), (105, 'M', 40, False, 'moderate'),
    (75, 'M', 40, False, 'severe'), (125, 'F', 40, False, 'norm'), (115, 'F', 40, False, 'mild'),
    (105, None, 3, False, 'mild'), (95, None, 3, False, 'moderate'), (65, None, 3, False, 'severe'),
    (100, 'F', 30, True, 'mild'), (110, 'F', 30, True, 'norm')])
def test_who_grade(hb, sex, age, preg, grade):
    assert who_grade(hb, sex, age, preg) == grade


def test_boundaries_are_inclusive_on_normal_side():
    assert not is_anemic(130, 'M', 30) and is_anemic(129.9, 'M', 30)
