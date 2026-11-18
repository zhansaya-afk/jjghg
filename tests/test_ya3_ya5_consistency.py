"""Я3 и Я5 считают анализ крови по одним нормам — ВОЗ 2024 (с триместром и высотой, если указаны).
Если по ВОЗ 2024 категорию определить нельзя, Я5 так и говорит, а не подставляет старую норму."""
import builtins
import itertools
import json
import re
from pathlib import Path

import pytest

from anemia.labs import classify_who2024, lab_report
from anemia.personalizer.facts import build_facts, grade_of, norm_g_l

NB = Path(__file__).resolve().parents[1] / 'notebooks' / 'colab_train.ipynb'
SEX = {'Женский': 'F', 'Мужской': 'M'}
TRIM = {'I': 1, 'II': 2, 'III': 3}


def _cell(title):
    nb = json.loads(NB.read_text(encoding='utf-8'))
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code' and title in ''.join(c['source'])]
    assert len(cells) == 1
    return ''.join(cells[0]['source'])


def _run_ya3(**params):
    src = _cell('Ячейка Я3')
    for name, value in params.items():
        src, n = re.subn(rf'^{name} = .*$', f'{name} = {value!r}', src, flags=re.MULTILINE)
        assert n == 1, name
    ns = {}
    exec(compile(src, 'Я3', 'exec'), ns)  # noqa: S102 — код ячейки блокнота
    return ns


def _profile(sex, age, preg, trim='Не знаю', h=''):
    p = {'sex': SEX[sex], 'age': age, 'pregnant': (preg == 'Да') if sex == 'Женский' else None}
    if p['pregnant']:
        p['trimester'] = TRIM.get(trim)
    if h:
        p['elevation_m'] = h
    return p


GRID = [g for g in itertools.product(['6.9', '9.7', '10.4', '10.7', '11.5', '12.2', '12.5', '13.5'],
                                     ['Женский', 'Мужской'], [13, 30, 70], ['Нет', 'Да'],
                                     ['Не знаю', 'I', 'II', 'III'], ['', '550'])
        if not (g[1] == 'Мужской' and g[3] == 'Да') and not (g[3] == 'Нет' and g[4] != 'Не знаю')
        and not (g[2] == 13 and g[3] == 'Да')]


@pytest.mark.parametrize('hb,sex,age,preg,trim,h', GRID)
def test_ya5_lab_facts_match_ya3(hb, sex, age, preg, trim, h, capsys):
    ns = _run_ya3(Hb=hb, Единица='г/дл', Пол_анализ=sex, Возраст_анализ=age, Беременность_анализ=preg,
                  Триместр_анализ=trim, Высота_проживания_м=h)
    capsys.readouterr()
    prof = _profile(sex, age, preg, trim, h)
    who = classify_who2024(float(hb) * 10, prof['sex'], age, prof['pregnant'], prof.get('trimester'), h or None)
    lab = build_facts(lab=ns['lab'], profile=prof)['lab']
    assert lab['grade'] == who['grade']                                        # та же категория, что в Я3
    if who['status'] == 'ok':
        assert lab['normal_from_g_l'] == who['normal_from_g_l']
        assert norm_g_l({'profile': prof}) == round(who['normal_from_g_l'])    # норма в тексте Я5 = норма Я3
    else:
        assert lab['grade'] is None and 'триместр' in lab['message']
    # то же число из вопроса пересчитывается так же
    assert grade_of(float(hb) * 10, {'profile': prof}) == who['grade']


def _ya5(ns, monkeypatch, capsys):
    ya5 = _cell('Ячейка Я5').replace("Что_объяснить = 'Фото (Я2)'", "Что_объяснить = 'Анализ крови (Я3)'")
    ya5 = ya5.replace('Опрос = True', 'Опрос = False')
    monkeypatch.setattr(builtins, 'input', lambda prompt='': '')
    exec(compile(ya5, 'Я5', 'exec'), dict(ns))  # noqa: S102 — код ячейки блокнота
    return capsys.readouterr().out


def test_ya5_second_trimester_uses_105(monkeypatch, capsys):
    ns = _run_ya3(Hb='10.7', Единица='г/дл', Пол_анализ='Женский', Возраст_анализ=28, Беременность_анализ='Да',
                  Триместр_анализ='II')
    capsys.readouterr()
    out = _ya5(ns, monkeypatch, capsys)
    assert '107 г/л' in out and 'от 105 г/л' in out and 'в пределах нормы' in out


def test_ya5_unknown_trimester_says_undetermined(monkeypatch, capsys):
    ns = _run_ya3(Hb='10.7', Единица='г/дл', Пол_анализ='Женский', Возраст_анализ=28, Беременность_анализ='Да',
                  Триместр_анализ='Не знаю')
    capsys.readouterr()
    assert ns['lab']['grade'] is None
    out = _ya5(ns, monkeypatch, capsys)
    assert '107 г/л' in out and 'категорию нельзя определить' in out and 'Укажите триместр' in out
    assert 'Покажите анализ врачу, который ведёт беременность' in out  # направление к врачу не теряется


def test_unknown_pregnancy_wording_and_referral():
    from anemia.personalizer import answer

    facts = build_facts(profile={'sex': 'F', 'age': 30, 'pregnant': None})
    out = answer('гемоглабин 9,5 это норм?', facts)['text']
    assert 'не указано, есть ли беременность' in out and 'триместр' in out
    assert 'Покажите анализ' in out


def test_ya5_altitude_norm_matches_ya3(monkeypatch, capsys):
    ns = _run_ya3(Hb='12.2', Единица='г/дл', Пол_анализ='Женский', Возраст_анализ=30, Беременность_анализ='Нет',
                  Триместр_анализ='Не знаю', Высота_проживания_м='550')
    capsys.readouterr()
    out = _ya5(ns, monkeypatch, capsys)
    assert '122 г/л' in out and 'от 124 г/л' in out and 'ниже нормы' in out


def test_contradiction_in_ya3_not_passed_to_ya5(capsys):
    ns = _run_ya3(Hb='12.2', Единица='г/дл', Пол_анализ='Женский', Возраст_анализ=30, Беременность_анализ='Нет',
                  Триместр_анализ='II')
    assert ns['lab'] is None


def test_adults_same_as_before():
    """Небеременные взрослые: ВОЗ 2024 = ВОЗ 2011, ответы Я5 для них не меняются."""
    from anemia.labs import lab_hb_report
    for hb, sex in itertools.product(range(60, 171, 3), ['F', 'M']):
        new, old = lab_report(hb, 'г/л', sex, 35, False), lab_hb_report(hb, 'г/л', sex, 35, False)
        assert (new['grade'], new['normal_from_g_l']) == (old['grade'], old['normal_from_g_l'])


def test_high_altitude_says_why_and_no_unadjusted_claim():
    from anemia.personalizer import answer

    facts = build_facts(profile={'sex': 'F', 'age': 30, 'pregnant': False, 'elevation_m': 3500})
    out = answer('гемоглобин 118 это норм?', facts)['text']
    assert 'от 3000 м' in out and 'Покажите анализ' in out
    assert 'в пределах нормы' not in out and 'ниже нормы' not in out


def test_pregnancy_numbers_not_allowed_for_men():
    from anemia.personalizer_eval import invented_numbers

    men = {'profile': {'sex': 'M', 'age': 40, 'pregnant': False}}
    women = {'profile': {'sex': 'F', 'age': 30, 'pregnant': None}}
    assert invented_numbers('порог 105 г/л', men) and not invented_numbers('порог 105 г/л', women)
