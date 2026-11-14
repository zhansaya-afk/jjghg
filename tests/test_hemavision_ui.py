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


def test_make_input_passes_result_fields_without_photo():
    """В персонализатор уходят только поля результата: ни фото, ни вырезок, ни масок; без оценки — без P и ткани."""
    import dataclasses

    ok = {'status': 'ok', 'stage': 'ok', 'risk': 'высокий', 'p_anemia': np.float64(0.83), 'hb_range': [92.0, 112.0],
          'tissues': ['conj'], 'issues': [], 'crop': np.zeros((4, 4, 3)), 'conj': np.ones((4, 4), bool)}
    inp = P.make_input(ok, V.Survey(sex='F', age=30, wellbeing={'weakness': True}))
    assert (inp.stage, inp.p_anemia, inp.tissues, inp.issues) == ('ok', 0.83, ('conj',), ())
    for v in dataclasses.asdict(inp).values():
        assert not isinstance(v, np.ndarray)
    bad = P.make_input(_refused_dict(tissues=['conj']), V.Survey(sex='F', age=30))
    assert bad.p_anemia is None and bad.tissues == () and bad.stage == 'ood'
    assert bad.issues == ('снимок <b>не похож</b> на обучающие',)


def test_rules_personalizer_explains_without_changing_result():
    import dataclasses

    from anemia.personalizer import build_facts
    from anemia.personalizer_eval import violations

    rp = P.RulesPersonalizer()
    assert isinstance(rp, P.Personalizer)
    inp = P.PersonalizerInput(status='ok', risk='высокий', hb_range=(92.0, 112.0), sex='F', age=30, pregnant=False,
                              wellbeing={'weakness': True, 'blood_loss': None, 'meat_fish': False}, stage='ok',
                              p_anemia=0.83, tissues=('conj',))
    rep = rp.reply(inp)
    assert rep.available and rep.source == 'rules' and 'высок' in rep.text.lower()
    facts = build_facts({'status': 'ok', 'risk': 'высокий', 'p_anemia': 0.83, 'hb_range': [92, 112],
                         'tissues': ['conj']}, None, {'sex': 'F', 'age': 30, 'pregnant': False})
    assert violations(rep.text, facts) == {}
    retake = rp.reply(P.make_input(_refused_dict(status='retake', stage='frame', issues=['снимок тёмный']),
                                   V.Survey(sex='F', age=30)))
    assert retake.available and 'высок' not in retake.text.lower() and 'г/л' not in retake.text
    child = rp.reply(P.make_input(_refused_dict(stage='age', issues=['младше 5 лет']), V.Survey(sex='F', age=3)))
    assert child.title.endswith('детям до 5 лет')
    urgent = rp.reply(dataclasses.replace(inp, question='Болит в сердце и тяжело дышать'))
    assert 'срочно' in urgent.title


def test_rules_personalizer_works_offline(monkeypatch):
    """Без сети и без языковых моделей: любое сетевое соединение → ошибка теста."""
    import socket
    import sys

    def no_network(*a, **kw):
        raise AssertionError('персонализатор пытался выйти в сеть')

    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(socket, 'create_connection', no_network)
    rep = P.RulesPersonalizer().reply(P.PersonalizerInput(status='ok', risk='низкий', hb_range=(120.0, 140.0),
                                                          sex='M', age=40, pregnant=None, stage='ok',
                                                          question='Что значит мой результат?'))
    assert rep.available and rep.source == 'rules'
    assert 'openai' not in sys.modules and 'llama_cpp' not in sys.modules


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


# ------------------------------------------------------------------ редизайн: технические цифры только для врача
def _ratio(a, b):
    def lum(h):
        c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        c = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in c]
        return 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_palette_contrast():
    """WCAG AA: обычный текст ≥ 4,5:1 — основные пары цветов темы и уровней риска."""
    from app.hemavision import theme as T
    pairs = [(T.INK, '#ffffff'), (T.MUTED, '#ffffff'), (T.MUTED, T.BG), ('#ffffff', T.ACCENT),
             ('#ffffff', T.ACCENT_DARK), ('#166534', '#f0fdf4'), ('#92400e', '#fffbeb'), ('#b91c1c', '#fef2f2'),
             ('#334155', '#f1f5f9'), ('#ffffff', '#166534'), ('#ffffff', '#92400e'), ('#ffffff', '#b91c1c')]
    for fg, bg in pairs:
        assert _ratio(fg, bg) >= 4.5, (fg, bg, round(_ratio(fg, bg), 2))


def test_user_cards_have_no_technical_numbers_doctor_section_has_disclaimer(backend, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'})
    r, res = V.run_analysis(_jpeg(synthetic_eye()[0]), V.Survey(sex='F', age=30), backend)
    assert r['status'] == 'ok'
    user = V.risk_card_html(r) + V.hb_html(r) + V.summary_html(r, res) + V.quality_html(res)
    for tech in ('P(анемия)', 'MAE', 'CV', 'Platt', 'порог', 'AUC'):
        assert tech not in user, tech
    doc = V.doctor_html(r, backend.version)
    assert 'не клиническая валидация' in doc and 'P(анемия)' in doc and backend.version in doc
    assert V.CV_NOTE in V.report_text(r, V.Survey(), backend.version)


def test_states_markup():
    assert 'role="status"' in V.loading_html() and 'Идёт анализ' in V.loading_html()
    err = V.error_html(ValueError('сбой'))
    assert 'РИСК НЕ ОПРЕДЕЛЁН' in err and 'сбой' in err and 'г/л' not in err
    assert V.legend_html(False, False) == '' and 'голубой' in V.legend_html(True, False)
    assert 'Готово к работе' in V.model_status_html(V.Backend(version='v'))
    for name in V.ICONS:
        assert V.icon(name).startswith('<span class="hv-ic" aria-hidden="true"><svg')


# ------------------------------------------------------------------ готовый анализ крови (как Я3)
def test_parse_lab_units_validation_and_empty():
    f = V.Survey(sex='F', age=30)
    assert V.parse_lab('', 'г/л', f) == (None, None) and V.parse_lab(None, 'г/л', f) == (None, None)
    assert V.parse_lab('   ', 'г/дл', f) == (None, None)
    for value, unit in (('125', 'г/л'), ('12,5', 'г/дл'), ('12.5', 'г/дл'), ('7,76', 'ммоль/л')):
        lab, err = V.parse_lab(value, unit, f)
        assert err is None and abs(lab['hb_g_l'] - 125) < 0.5 and lab['source'] == 'лабораторный анализ'
    for value, unit in (('abc', 'г/л'), ('12,5', 'г/л'), ('125', 'г/дл'), ('12', None), ('125', 'мг')):
        lab, err = V.parse_lab(value, unit, f)
        assert lab is None and err, (value, unit)
    # нормы ВОЗ по той же анкете, что и фото
    assert V.parse_lab('125', 'г/л', f)[0]['grade'] == 'norm'
    assert V.parse_lab('125', 'г/л', V.Survey(sex='M', age=30))[0]['grade'] == 'mild'
    assert V.parse_lab('112', 'г/л', V.Survey(sex='F', age=30, pregnant=True))[0]['grade'] == 'norm'


def test_lab_card_is_separate_from_photo_hb(backend, monkeypatch):
    _route(monkeypatch, {'conj', 'sclera'})
    survey = V.parse_survey('Женский', 30, 'Нет', 'Авто')
    r, _ = V.run_analysis(_jpeg(synthetic_eye()[0]), survey, backend)
    lab, _ = V.parse_lab('11,8', 'г/дл', survey)
    card = V.lab_html(lab)
    assert 'анализа крови' in card and 'лабораторный' in card and '118' in card and 'не меняет' in card
    assert 'экспериментально' not in card and 'по фото' in card       # «отдельно от оценки по фото»
    hb = V.hb_html(r)
    assert 'лабораторный' not in hb and ('по фото' in hb or 'По этому фото' in hb)
    assert V.lab_html(None) == ''
    bad = V.lab_html(None, 'неизвестная единица Hb')
    assert 'не учтён' in bad and 'неизвестная единица' in bad
    rep = V.report_text(r, survey, backend.version, lab)
    photo_part, lab_part = rep.split('Анализ крови (введён пользователем')
    assert 'Результат по фото:' in photo_part and PR.format_result(r) in photo_part and '118 г/л' not in photo_part
    assert lab['text'] in lab_part
    assert 'не введён' in V.report_text(r, survey, backend.version).split('Анализ крови (введён')[1]


def test_lab_goes_to_personalizer_but_not_into_photo_fields():
    ok = {'status': 'ok', 'stage': 'ok', 'risk': 'высокий', 'p_anemia': 0.83, 'hb_range': [92.0, 112.0],
          'tissues': ['conj'], 'issues': []}
    survey = V.Survey(sex='F', age=30)
    lab, _ = V.parse_lab('11,8', 'г/дл', survey)
    a, b = P.make_input(ok, survey), P.make_input(ok, survey, lab)
    assert (a.risk, a.p_anemia, a.hb_range) == (b.risk, b.p_anemia, b.hb_range) == ('высокий', 0.83, (92.0, 112.0))
    assert a.lab is None and b.lab is lab
    text = P.RulesPersonalizer().reply(b).text
    assert 'риск анемии высокий' in text and '92–112 г/л' in text and '118 г/л' in text
    # снимок не подошёл: про фото — «переснимите», анализ крови всё равно объясняется
    refused = P.make_input(_refused_dict(status='retake', stage='frame', issues=['снимок тёмный']), survey, lab)
    assert refused.risk is None and refused.hb_range is None and refused.lab is lab
    text = P.RulesPersonalizer().reply(refused).text
    assert '118 г/л' in text and '80' not in text and 'высок' not in text.lower()


# ------------------------------------------------------------------ диалог с персонализатором (как Я5)
OK_INPUT = P.PersonalizerInput(status='ok', risk='высокий', hb_range=(92.0, 112.0), sex='F', age=30, pregnant=False,
                               wellbeing={'weakness': True}, stage='ok', p_anemia=0.83, tissues=('conj',))


def _dialog(questions, inp=OK_INPUT, personalizer=None):
    from anemia.personalizer import build_facts
    from anemia.personalizer_eval import violations

    rp = personalizer or P.RulesPersonalizer()
    first = rp.reply(inp)
    history, replies = P.start_history(first), []
    facts = build_facts({'status': inp.status, 'stage': inp.stage, 'risk': inp.risk, 'p_anemia': inp.p_anemia,
                         'hb_range': list(inp.hb_range or ()), 'tissues': list(inp.tissues)}, inp.lab,
                        {'sex': inp.sex, 'age': inp.age, 'pregnant': inp.pregnant})
    for q in questions:
        rep, history = P.ask(rp, inp, history, q)
        replies.append(rep)
        assert rep is None or violations(rep.text, facts, q) == {}, (q, rep.text)
    return first, replies, history


def test_dialog_keeps_session_context():
    first, (a, b), history = _dialog(['Что такое ферритин?', 'А для мужчин?'])
    assert 'Ферритин' in a.text and 'Ферритин' in b.text          # уточнение понято по прошлому вопросу
    assert [m['role'] for m in history] == ['user', 'assistant'] * 3 and history[1]['text'] == first.text
    msgs = P.chat_messages(history)                                # объяснение — карточкой, в чате только вопросы
    assert [m['content'] for m in msgs[::2]] == ['Что такое ферритин?', 'А для мужчин?']
    assert all(set(m) == {'role', 'content'} for m in msgs)


def test_dialog_russian_and_kazakh_with_safety_rules():
    _, (ru, kk, dose, urgent_kk, urgent_ru), _ = _dialog([
        'Что значит мой результат?', 'Нәтижем нені білдіреді?', 'Сколько таблеток железа пить?',
        'Жүрегім ауырады, тыныс алу қиын', 'Болит в сердце и тяжело дышать'])
    assert 'риск анемии высокий' in ru.text
    assert 'қаупі жоғары' in kk.text and 'диагноз емес' in kk.text   # ответ на казахском: риск из результата
    assert 'дозу подбирает врач' in dose.text and not any(ch.isdigit() for ch in dose.text)
    for rep in (urgent_kk, urgent_ru):
        assert 'срочно' in rep.title and '103' in rep.text
    assert 'Бүгін дәрігерге' in urgent_kk.text


def test_dialog_empty_question_limit_and_engine_error(monkeypatch):
    rp = P.RulesPersonalizer()
    history = P.start_history(rp.reply(OK_INPUT))
    assert P.ask(rp, OK_INPUT, history, '   ') == (None, history)
    monkeypatch.setattr(P, 'MAX_TURNS', 2)
    for q in ('Что такое ферритин?', 'Что такое анемия?'):
        rep, history = P.ask(rp, OK_INPUT, history, q)
        assert rep.available
    rep, after = P.ask(rp, OK_INPUT, history, 'Что есть?')
    assert not rep.available and rep.text == P.LIMIT_TEXT and after == history
    long_q = 'анемия ' * 200
    _, h = P.ask(rp, OK_INPUT, P.start_history(rp.reply(OK_INPUT)), long_q)
    assert len(h[-2]['text']) <= P.MAX_QUESTION

    def boom(*a, **kw):
        raise RuntimeError('сбой базы')

    import anemia.personalizer as AP
    monkeypatch.setattr(AP, 'personalize', boom)
    history = P.start_history(P.PersonalizerReply('t', 'объяснение', True, 'rules'))
    rep, after = P.ask(rp, OK_INPUT, history, 'Что такое ферритин?')
    assert not rep.available and 'недоступно' in rep.text and after == history
    assert P.start_history(P.StubPersonalizer().reply(OK_INPUT)) == ()


def test_dialog_works_offline(monkeypatch):
    import socket
    import sys

    def no_network(*a, **kw):
        raise AssertionError('диалог пытался выйти в сеть')

    monkeypatch.setattr(socket.socket, 'connect', no_network)
    monkeypatch.setattr(socket, 'create_connection', no_network)
    _, replies, _ = _dialog(['Что такое ферритин?', 'Нәтижем нені білдіреді?'])
    assert all(r.available for r in replies)
    assert 'openai' not in sys.modules and 'llama_cpp' not in sys.modules


# ------------------------------------------------------------------ весь путь в собранном интерфейсе
def _fn(demo, name):
    return next(f.fn for f in demo.fns.values() if getattr(f.fn, '__name__', '') == name)


def test_full_path_photo_result_lab_personalizer(backend, monkeypatch):
    """Фото → анализ → результат → анализ крови → объяснение → диалог: через функции собранного интерфейса."""
    gr = pytest.importorskip('gradio')
    from app.hemavision import ui

    _route(monkeypatch, {'conj', 'sclera'})
    demo = ui.build(backend=backend)
    import tempfile

    from PIL import Image

    photo = synthetic_eye()[0]
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        Image.fromarray((np.clip(photo, 0, 1) * 255).astype(np.uint8)).save(f, format='JPEG', quality=95)
        path = f.name
    form = (path, 'Женский', 30, 'Нет')
    with pytest.raises(gr.Error, match='Анализ крови'):
        _fn(demo, 'check_lab')(*form[1:], '12,5', 'г/л')        # 12,5 г/л неправдоподобно → исправить на месте
    assert _fn(demo, 'check_lab')(*form[1:], '12,5', 'г/дл') == '' and _fn(demo, 'check_lab')(*form[1:], '', 'г/л') == ''
    _fn(demo, 'on_analyze')(path)
    out = _fn(demo, 'run')(*form, 'Авто', '12,5', 'г/дл', 'Да', 'Нет', 'Да')
    (_, _, _, _, risk_card, hb_card, lab_card, _, _, p_card, _, report, state, chat, chatbot, q, *_) = out
    r = state['r']
    assert r['status'] == 'ok' and V.LEVEL[r['risk']][2] in risk_card
    assert state['lab']['hb_g_l'] == 125.0 and 'лабораторный' in lab_card['value'] and lab_card['visible']
    assert 'лабораторный' not in hb_card['value']
    assert 'AI Personalizer' in p_card and '125 г/л' in p_card and chatbot == [] and q == ''
    with open(report['value'], encoding='utf-8') as f:
        assert '125 г/л' in f.read().split('Анализ крови (введён пользователем')[1]
    msgs, q, chat = _fn(demo, 'on_ask')('Что такое ферритин?', chat)
    assert q == '' and msgs[0]['content'] == 'Что такое ферритин?' and 'Ферритин' in msgs[1]['content']
    msgs, _, chat = _fn(demo, 'on_ask')('Нәтижем нені білдіреді?', chat)
    assert len(msgs) == 4 and 'диагноз емес' in msgs[3]['content']
    assert chat['inp'].risk == r['risk'] and chat['inp'].lab == state['lab']    # диалог не меняет результат
    # без анализа крови: карточки нет, путь тот же
    out = _fn(demo, 'run')(*form, 'Авто', '', 'г/л', 'Не знаю', 'Не знаю', 'Не знаю')
    assert out[6]['visible'] is False and out[12]['lab'] is None
