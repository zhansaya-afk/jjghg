"""Формулировка Hb по фото: «оценка ~105 г/л (точно не определить: 90–120), это НЕ анализ крови».
Меняется только текст: диапазон hb_range и прогноз не пересчитываются."""
from anemia import predict as PR
from anemia.personalizer import facts as F
from anemia.personalizer import texts as T


def test_hb_text_wording():
    r = {'hb_range': (90.0, 120.0), 'hb_estimate': 104.2}
    assert PR.hb_text(r) == 'оценка ~105 г/л (точно не определить: 90–120), это НЕ анализ крови'


def test_estimate_rounded_to_5_and_inside_range():
    assert PR.hb_estimate_shown({'hb_range': (90.0, 120.0), 'hb_estimate': 102.5}) == 105   # не банковское округление
    assert PR.hb_estimate_shown({'hb_range': (90.0, 120.0), 'hb_estimate': 102.4}) == 100
    assert PR.hb_estimate_shown({'hb_range': (104.0, 126.0), 'hb_estimate': 101.0}) == 104  # не выходит за диапазон


def test_old_dict_without_estimate_uses_midpoint():
    assert PR.hb_estimate_shown({'hb_range': (90.0, 120.0)}) == 105
    assert PR.hb_estimate_shown({'hb_range': None, 'hb_estimate': 104.0}) is None
    assert PR.hb_text({'hb_range': None}) == ''


def test_format_result_uses_new_wording():
    r = {'status': 'ok', 'model': 'm', 'kind': None, 'risk': 'средний', 'hb_range': (90.0, 120.0),
         'hb_estimate': 106.0, 'hb_note': 'экспериментально', 'reasons': [], 'notes': [], 'issues': [], 'disclaimer': PR.DISCLAIMER}
    txt = PR.format_result(r)
    assert 'Hb по фото: оценка ~105 г/л (точно не определить: 90–120), это НЕ анализ крови' in txt
    assert 'Ориентировочный Hb' not in txt


def test_personalizer_same_numbers_as_screen():
    raw = {'status': 'ok', 'risk': 'средний', 'p_anemia': 0.5, 'hb_range': (90.0, 120.0), 'hb_estimate': 104.0,
           'tissues': ['conj']}
    ph = F.photo_facts(raw)
    assert ph['hb_estimate_g_l'] == PR.hb_estimate_shown(raw) == 105
    line = T.hb_photo({'photo': ph})
    assert '~105 г/л' in line and '90–120' in line and 'НЕ анализ крови' in line
    assert F.photo_facts({**raw, 'status': 'retake'})['hb_estimate_g_l'] is None


def test_personalizer_card_without_estimate_invents_no_number():
    line = T.hb_photo({'photo': {'status': 'ok', 'hb_interval_g_l': [92, 112]}})
    assert '92–112 г/л' in line and 'НЕ анализ крови' in line and '~' not in line and '102' not in line


# ------------------------------------------------------------------ оценка < 100: пользователю без числа
LOW = {'status': 'ok', 'model': 'm', 'kind': None, 'risk': 'высокий', 'hb_range': (75.0, 120.0), 'hb_estimate': 96.0,
       'hb_note': 'экспериментально', 'reasons': [], 'notes': [], 'issues': [], 'disclaimer': PR.DISCLAIMER}


def test_low_threshold_same_in_core_and_personalizer():
    assert F.HB_LOW_SHOWN == PR.HB_LOW_SHOWN == 100


def test_low_estimate_detected_on_shown_value():
    assert PR.hb_low(LOW)
    assert not PR.hb_low({**LOW, 'hb_estimate': 97.5})     # показ ~100 → не низкая
    assert not PR.hb_low({**LOW, 'hb_range': None})


def test_low_estimate_doctor_text_keeps_number_with_caveat():
    txt = PR.format_result(LOW)
    assert 'оценка ~95 г/л (точно не определить: 75–120)' in txt and PR.HB_LOW_CAVEAT in txt
    assert PR.HB_LOW_CAVEAT not in PR.format_result({**LOW, 'hb_estimate': 110.0})


def test_low_estimate_hidden_from_personalizer():
    ph = F.photo_facts({**LOW, 'p_anemia': 0.8, 'tissues': ['conj']})
    assert ph['hb_low_hidden'] and ph['hb_interval_g_l'] is None and ph['hb_estimate_g_l'] is None
    line = T.hb_photo({'photo': ph})
    assert 'не показываем' in line and not any(c.isdigit() for c in line)
    direct = T.hb_photo({'photo': {'status': 'ok', 'hb_interval_g_l': [75, 115]}})   # карточка без оценки
    assert 'не показываем' in direct and '75' not in direct


def test_low_estimate_hidden_on_screen():
    from app.hemavision import view as V
    card = V.hb_html(LOW)
    assert 'не показываем' in card and 'г/л' not in card and '95' not in card and '75' not in card
    assert 'экспериментально' not in V.doctor_html(LOW)       # раздел на экране: цифр Hb тоже нет
    ok = V.hb_html({**LOW, 'hb_estimate': 110.0})
    assert '~110' in ok and 'Точно не определить: 75–120' in ok and 'НЕ анализ крови' in ok
