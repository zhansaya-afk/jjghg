"""Интерфейс HemaVision (app/hemavision): отображение результата ядра и заглушка AI Personalizer.
Синтетика и малая модель из test_api — проверяются медицинские правила показа, не точность."""
import html

import numpy as np
import pytest
from test_api import _jpeg, _route, models  # noqa: F401  (models — фикстура)
from test_regions import synthetic_eye

from anemia import core as CORE
from anemia import predict as PR
from app.hemavision import personalizer as P
from app.hemavision import view as V


@pytest.fixture
def backend(models):  # noqa: F811
    return V.Backend(bundle=models.bundle, landmarker=None, version=models.version)


def _refused_dict(**kw):
    r = {'status': 'unreliable', 'stage': 'ood', 'risk': 'высокий', 'p_anemia': 0.9, 'hb_range': (80.0, 100.0),
         'hb_note': 'экспериментально', 'issues': ['снимок <b>не похож</b> на обучающие'], 'notes': [],
         'reasons': ['P(анемия) = 0.90'], 'model': 'm'}
    r.update(kw)
    return r


def _no_numbers(*htmls):
    txt = ' '.join(htmls)
    assert 'г/л' not in txt and 'P(анемия)' not in txt
    for lv in ('НИЗКИЙ РИСК', 'СРЕДНИЙ РИСК', 'ВЫСОКИЙ РИСК'):
        assert lv not in txt


# ------------------------------------------------------------------ анкета
def test_parse_survey():
    s = V.parse_survey('Мужской', 40, 'Да', 'Авто', ('Да', 'Нет', 'Не знаю'))
    assert s.sex == 'M' and s.age == 40.0 and s.pregnant is None and s.method is None
    assert s.wellbeing == {'weakness': True, 'blood_loss': False, 'meat_fish': None}
    s = V.parse_survey('Женский', None, 'Да', 'Склера')
    assert s.sex == 'F' and s.age is None and s.pregnant is True and s.method == 'sclera' and s.wellbeing == {}
    assert V.parse_survey('Не указан', '', 'Нет', 'Обе ткани').method == 'both'
    assert set(V.METHOD.values()) - {None} == set(PR.METHODS)


# ------------------------------------------------------------------ нормальное фото
def test_ok_result_shows_level_in_words_and_scale(backend, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'})
    r, res = V.run_analysis(_jpeg(synthetic_eye()[0]), V.parse_survey('Женский', 30, 'Нет', 'Авто'), backend)
    assert r['status'] == 'ok' and r['risk'] in CORE.RISK_LEVELS
    card = V.risk_card_html(r)
    assert V.LEVEL[r['risk']][2] in card                       # уровень словом, не только цветом
    assert html.escape(PR.RISK_TEXT[r['risk']][1:]) in card     # пояснение ядра
    assert card.count('ваш результат') == 1
    assert all(st == 'done' for _, st in V.step_states(r))
    hb = V.hb_html(r)
    if r['hb_range']:
        assert 'экспериментально' in hb and 'г/л' in hb and 'анализ крови' in hb
    else:
        assert hb and 'г/л' not in hb.split('MAE')[0]
    q = V.quality_html(res)
    assert 'в норме' in q and 'обе' in q
    crop, face = V.overlays(r)
    assert crop is not None and crop.dtype == np.uint8 and face is None
    rep = V.report_text(r, V.parse_survey('Женский', 30, 'Нет', 'Авто', ('Да', 'Нет', 'Да')), backend.version)
    assert PR.DISCLAIMER in rep and 'не влияют' in rep and 'возраст: 30' in rep and 'РИСК АНЕМИИ' in rep


# ------------------------------------------------------------------ отказы: без уровня и без Hb
def test_ood_refusal_has_no_level_no_hb(backend, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'})
    monkeypatch.setattr(CORE, 'ood_share', lambda x, lo, hi: np.ones(len(x)))
    r, res = V.run_analysis(synthetic_eye()[0], V.Survey(sex='F', age=30), backend)
    assert r['status'] == 'unreliable' and r['stage'] == 'ood'
    card, hb = V.risk_card_html(r), V.hb_html(r)
    assert 'РИСК НЕ ОПРЕДЕЛЁН' in card and 'анализ крови' in card and hb == ''
    _no_numbers(card, hb)
    assert [st for _, st in V.step_states(r)] == ['done'] * 5 + ['fail']
    assert 'Ориентировочный Hb' not in V.report_text(r, V.Survey(), backend.version)


def test_child_refused_before_photo(backend):
    r, _ = V.run_analysis(synthetic_eye()[0], V.Survey(sex='F', age=3), backend)
    assert r['stage'] == 'age'
    assert [st for _, st in V.step_states(r)] == ['fail'] + ['skip'] * 5
    card = V.risk_card_html(r)
    assert html.escape(PR.CHILD_REFUSAL) in card
    _no_numbers(card, V.hb_html(r))


def test_dark_frame_stops_at_frame(backend):
    r, res = V.run_analysis(synthetic_eye()[0] * 0.05, V.Survey(sex='M', age=40), backend)
    assert r['status'] == 'retake' and r['stage'] == 'frame'
    assert [st for _, st in V.step_states(r)][:3] == ['done', 'fail', 'skip']
    assert 'темно' in V.quality_html(res)
    assert V.overlays(r) == (None, None)


def test_tissue_switch_is_passed_to_core(backend, monkeypatch):
    _route(monkeypatch, {'conj'})
    r, _ = V.run_analysis(synthetic_eye()[0], V.parse_survey('Женский', 30, 'Нет', 'Склера'), backend)
    assert r['status'] != 'ok' and r['method'] == 'sclera' and 'другой метод не подставляется' in ' '.join(r['issues'])
    _no_numbers(V.risk_card_html(r), V.hb_html(r))


def test_refused_dict_hides_hb_even_if_present():
    """Защита на уровне показа: status ≠ 'ok' → ни уровня, ни P, ни Hb, даже если поля заполнены."""
    r = _refused_dict()
    card = V.risk_card_html(r)
    _no_numbers(card, V.hb_html(r), V.reasons_md(r))
    assert '<b>не похож</b>' not in card and '&lt;b&gt;не похож&lt;/b&gt;' in card    # текст экранирован
    inp = P.make_input(r, V.Survey(sex='F', age=30))
    assert inp.risk is None and inp.hb_range is None


# ------------------------------------------------------------------ AI Personalizer
def test_stub_personalizer_gives_no_advice():
    stub = P.StubPersonalizer()
    assert isinstance(stub, P.Personalizer)
    for wb in ({}, {'weakness': True, 'blood_loss': None, 'meat_fish': False}):
        rep = stub.reply(P.PersonalizerInput(status='ok', risk='высокий', hb_range=(90, 110), sex='F', age=30,
                                             pregnant=None, wellbeing=wb))
        assert rep.available is False and rep.source == 'stub'
        assert 'не влияют' in rep.text
        for word in ('диагноз', 'железо', 'препарат', 'мг', 'г/л', 'высокий'):
            assert word not in rep.text.lower()


def test_wellbeing_keys_match_personalizer_survey():
    assert [k for k, _ in P.WELLBEING] == ['weakness', 'blood_loss', 'meat_fish']


# ------------------------------------------------------------------ модель и сборка интерфейса
def test_load_backend_missing_model(tmp_path):
    b = V.load_backend(tmp_path / 'нет.joblib', landmarker=None)
    assert b.bundle is None and 'нет модели' in b.error
    assert 'Я1' in V.model_status_html(b)


def test_stepper():
    assert V.stepper_html(0) == ''
    s = V.stepper_html(3)
    assert s.count('class="done"') == 2 and s.count('class="cur"') == 1 and 'Шаг 3 из 4' in s


def test_build_ui(backend):
    gr = pytest.importorskip('gradio')
    from app.hemavision import ui

    for b in (backend, V.Backend(error='нет модели')):
        demo = ui.build(backend=b)
        assert isinstance(demo, gr.Blocks)
