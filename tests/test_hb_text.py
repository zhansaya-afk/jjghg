"""Показ Hb по фото (решение команды 2026-11-19). Меняется только текст, не расчёт:
- врачу (Я2, серия, отчёт .txt): «оценка ~105 г/л (точно не определить: 90–120), это НЕ анализ крови» + оговорка о
  точности всегда;
- пользователю (экран HemaVision, Personalizer ru/kk): числа Hb по фото нет никогда."""
import pytest

from anemia import predict as PR
from anemia.personalizer import facts as F
from anemia.personalizer import texts as T
from anemia.personalizer import texts_kk as TK

OK = {'status': 'ok', 'model': 'm', 'kind': None, 'risk': 'средний', 'hb_range': (90.0, 120.0), 'hb_estimate': 104.2,
      'hb_note': 'экспериментально', 'reasons': [], 'notes': [], 'issues': [], 'disclaimer': PR.DISCLAIMER,
      'p_anemia': 0.5, 'tissues': ['conj']}


def test_hb_text_wording():
    assert PR.hb_text(OK) == 'оценка ~105 г/л (точно не определить: 90–120), это НЕ анализ крови'


def test_estimate_rounded_to_5_and_inside_range():
    assert PR.hb_estimate_shown({'hb_range': (90.0, 120.0), 'hb_estimate': 102.5}) == 105   # не банковское округление
    assert PR.hb_estimate_shown({'hb_range': (90.0, 120.0), 'hb_estimate': 102.4}) == 100
    assert PR.hb_estimate_shown({'hb_range': (104.0, 126.0), 'hb_estimate': 101.0}) == 104  # не выходит за диапазон


def test_old_dict_without_estimate_uses_midpoint():
    assert PR.hb_estimate_shown({'hb_range': (90.0, 120.0)}) == 105
    assert PR.hb_estimate_shown({'hb_range': None, 'hb_estimate': 104.0}) is None
    assert PR.hb_text({'hb_range': None}) == ''


@pytest.mark.parametrize('est', [80.0, 99.0, 104.2, 140.0])
def test_doctor_text_always_has_number_and_caveat(est):
    txt = PR.format_result({**OK, 'hb_estimate': est})
    assert 'Hb по фото: оценка ~' in txt and 'точно не определить: 90–120' in txt and 'НЕ анализ крови' in txt
    assert PR.HB_DOCTOR_CAVEAT in txt and 'Ориентировочный Hb' not in txt


def test_doctor_caveat_has_no_wrong_low_claim():
    # «при ОЦЕНКЕ ниже 100 завышено» — неверно (у таких оценок модель в среднем занижает); речь о РЕАЛЬНОМ Hb
    assert 'при реальном Hb < 100' in PR.HB_DOCTOR_CAVEAT and 'оценка ниже 100' not in PR.HB_DOCTOR_CAVEAT


def test_no_caveat_without_hb():
    txt = PR.format_result({**OK, 'hb_range': None, 'hb_note': 'цифра Hb не выдаётся'})
    assert PR.HB_DOCTOR_CAVEAT not in txt


@pytest.mark.parametrize('est', [80.0, 104.2, 140.0])
def test_screen_never_shows_photo_hb_number(est):
    from app.hemavision import view as V
    r = {**OK, 'hb_estimate': est}
    card = V.hb_html(r)
    assert 'не показываем' in card and 'г/л' not in card and not any(x in card for x in ('90', '120', '105', '80'))
    assert 'экспериментально' not in V.doctor_html(r)


def test_personalizer_never_gets_photo_hb_number():
    ph = F.photo_facts(OK)
    assert ph['hb_hidden'] and ph['hb_interval_g_l'] is None and ph['hb_estimate_g_l'] is None
    for line in (T.hb_photo({'photo': ph}), TK.hb_photo({'photo': ph})):
        assert line and not any(c.isdigit() for c in line)
    direct = T.hb_photo({'photo': {'status': 'ok', 'hb_interval_g_l': [92, 112]}})   # карточка, собранная вручную
    assert 'не показываем' in direct and '92' not in direct
    assert not any(c.isdigit() for c in TK.hb_photo({'photo': {'status': 'ok', 'hb_interval_g_l': [92, 112]}}))
    assert 'hb_hidden' not in F.photo_facts({**OK, 'status': 'retake'})
