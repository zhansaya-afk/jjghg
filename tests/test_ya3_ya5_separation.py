"""Я3 показывает ВОЗ 2024 + шкалу РК, а в Я5 (персонализатор) уходит прежний отчёт lab_hb_report (ВОЗ 2011):
новая классификация не меняет ответы Я5."""
import builtins
import itertools
import json
import re
from pathlib import Path

import pytest

from anemia.labs import lab_hb_report
from anemia.personalizer.facts import build_facts

NB = Path(__file__).resolve().parents[1] / 'notebooks' / 'colab_train.ipynb'


def _cell(title):
    nb = json.loads(NB.read_text(encoding='utf-8'))
    cells = [c for c in nb['cells'] if c['cell_type'] == 'code' and title in ''.join(c['source'])]
    assert len(cells) == 1
    return ''.join(cells[0]['source'])


def _run_ya3(capsys=None, **params):
    """Код ячейки Я3 с заданными полями формы → пространство имён ноутбука (как globals() в Colab)."""
    src = _cell('Ячейка Я3')
    for name, value in params.items():
        src, n = re.subn(rf'^{name} = .*$', f'{name} = {value!r}', src, flags=re.MULTILINE)
        assert n == 1, name
    ns = {}
    exec(compile(src, 'Я3', 'exec'), ns)  # noqa: S102 — код ячейки блокнота
    return ns


def _old_ya3_lab(hb, unit, sex_ru, age, preg_ru):
    """Ячейка Я3 на main f686133 — дословно её вызов."""
    return lab_hb_report(hb, unit, sex={'Женский': 'F', 'Мужской': 'M'}.get(sex_ru), age=age,
                         pregnant=(preg_ru == 'Да') if sex_ru == 'Женский' else None)


GRID = list(itertools.product(['6.9', '9.5', '10.4', '10.7', '11.5', '12.5', '13.5'], ['Женский', 'Мужской'],
                              [1, 13, 30, 70], ['Нет', 'Да'], ['Не знаю', 'I', 'II', 'III']))


@pytest.mark.parametrize('hb,sex,age,preg,trim', [g for g in GRID if not (g[1] == 'Мужской' and g[3] == 'Да')])
def test_ya3_passes_old_who2011_report_to_ya5(hb, sex, age, preg, trim, capsys):
    ns = _run_ya3(Hb=hb, Единица='г/дл', Пол_анализ=sex, Возраст_анализ=age, Беременность_анализ=preg,
                  Триместр_анализ=trim)
    old = _old_ya3_lab(hb, 'г/дл', sex, age, preg)
    assert ns['lab'] == old                                              # Я5 получает ровно прежний отчёт
    profile = {'sex': {'Женский': 'F', 'Мужской': 'M'}[sex], 'age': age,
               'pregnant': (preg == 'Да') if sex == 'Женский' else None}
    assert build_facts(lab=ns['lab'], profile=profile) == build_facts(lab=old, profile=profile)
    out = capsys.readouterr().out
    assert 'ВОЗ 2024' in out and 'МЗ РК 2017' in out                     # Я3 показывает обе новые системы
    assert 'передаётся прежняя оценка ВОЗ 2011' in out


@pytest.mark.parametrize('hb,trim', [('10.7', 'II'), ('10.7', 'Не знаю'), ('9.7', 'Не знаю')])
def test_ya5_answer_same_where_who2024_differs_or_refuses(hb, trim, monkeypatch, capsys):
    """II триместр и неизвестный триместр: ВОЗ 2024 даёт другую категорию или отказ, а ответ Я5 — прежний."""
    ns = _run_ya3(Hb=hb, Единица='г/дл', Пол_анализ='Женский', Возраст_анализ=28, Беременность_анализ='Да',
                  Триместр_анализ=trim)
    capsys.readouterr()
    old_ns = {**ns, 'lab': _old_ya3_lab(hb, 'г/дл', 'Женский', 28, 'Да')}
    ya5 = _cell('Ячейка Я5').replace("Что_объяснить = 'Фото (Я2)'", "Что_объяснить = 'Анализ крови (Я3)'")
    ya5 = ya5.replace('Опрос = True', 'Опрос = False')
    outs = []
    for space in (ns, old_ns):
        monkeypatch.setattr(builtins, 'input', lambda prompt='': '')
        exec(compile(ya5, 'Я5', 'exec'), dict(space))  # noqa: S102 — код ячейки блокнота
        outs.append(capsys.readouterr().out)
    assert outs[0] == outs[1] and outs[0]


def test_ya3_new_report_kept_separately():
    ns = _run_ya3(Hb='10.7', Единица='г/дл', Пол_анализ='Женский', Возраст_анализ=28, Беременность_анализ='Да',
                  Триместр_анализ='II')
    assert ns['lab_2024']['who2024']['grade'] == 'norm' and ns['lab']['grade'] == 'mild'   # ВОЗ 2024 ≠ ВОЗ 2011
    ns = _run_ya3(Hb='10.7', Единица='г/дл', Пол_анализ='Женский', Возраст_анализ=28, Беременность_анализ='Да',
                  Триместр_анализ='Не знаю')
    assert ns['lab_2024'] is None and ns['lab']['grade'] == 'mild'       # отказ ВОЗ 2024 не лишает Я5 анализа
