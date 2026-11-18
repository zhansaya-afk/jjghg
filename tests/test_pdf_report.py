"""PDF-отчёт HemaVision (app/hemavision/pdf_report.py): полный, частичный результат и отказ.
Синтетика и малая модель из test_api — проверяются правила показа и кириллица, не точность."""
import dataclasses

import pytest
from test_api import _jpeg, _route, models  # noqa: F401  (models — фикстура)
from test_regions import synthetic_eye

from anemia import labs as LABS
from app.hemavision import personalizer as P
from app.hemavision import view as V

pytest.importorskip('reportlab')
pypdf = pytest.importorskip('pypdf')
from app.hemavision import pdf_report as R

LEVEL_WORDS = ('НИЗКИЙ РИСК', 'СРЕДНИЙ РИСК', 'ВЫСОКИЙ РИСК')


@pytest.fixture
def backend(models):  # noqa: F811
    return V.Backend(bundle=models.bundle, landmarker=None, version=models.version)


def _text(path):
    rd = pypdf.PdfReader(path)
    return ' '.join(' '.join(p.extract_text().split()) for p in rd.pages), rd


def _n_images(rd):
    return sum(len(p.images) for p in rd.pages)


def _pdf(tmp_path, r, survey, **kw):
    path = R.build_pdf(str(tmp_path / 'report.pdf'), r, survey, **kw)
    with open(path, 'rb') as f:
        assert f.read(5) == b'%PDF-'
    return _text(path)


def test_full_result_real_pipeline(tmp_path, backend, monkeypatch):
    """Полный путь: фото → analyze_photo → PDF с фото, маской, качеством, нормами, Personalizer и дисклеймером."""
    _route(monkeypatch, {'conj', 'sclera'})
    photo = _jpeg(synthetic_eye()[0])
    survey = V.parse_survey('Женский', 30, 'Нет', 'Авто', ('Да', 'Нет', 'Не знаю'))
    r, res = V.run_analysis(photo, survey, backend)
    assert r['status'] == 'ok'
    lab, err = V.parse_lab('11,8', 'г/дл', survey)
    assert err is None
    reply = P.RulesPersonalizer().reply(P.make_input(r, survey, lab))
    hist = P.start_history(reply)
    _, hist = P.ask(P.RulesPersonalizer(), P.make_input(r, survey, lab), hist, 'Что такое ферритин?')
    txt, rd = _pdf(tmp_path, r, survey, res=res, lab=lab, photo=photo, model_version=backend.version, reply=reply,
                   history=hist)
    assert f'{r["risk"].upper()} РИСК' in txt
    assert sum(w in txt for w in LEVEL_WORDS) == 1
    assert R.DISCLAIMER in txt and 'Качество фотографии' in txt and 'Резкость' in txt
    assert backend.version in txt and 'Дата и время' in txt
    assert 'ВОЗ 2011' in txt and 'норма Hb ≥ 120 г/л' in txt               # женщина 30 лет, ВОЗ 2011
    assert 'Вопрос: Что такое ферритин?' in txt and 'Ответов Personalizer нет' not in txt
    assert 'Анализ крови' in txt and '118 г/л' in txt
    assert _n_images(rd) >= 2                                                # исходное фото + маска
    if r['hb_range']:
        assert 'НЕ лабораторное измерение' in txt and 'Неопределённость' in txt
    else:
        assert 'Hb по фото: не показан' in txt


def test_hb_shown_only_from_core_with_uncertainty(tmp_path, backend, monkeypatch):
    """Hb печатается только из hb_range ядра — с его неопределённостью (hb_note) и пометкой «не лабораторное»."""
    _route(monkeypatch, {'conj', 'sclera'})
    r, res = V.run_analysis(_jpeg(synthetic_eye()[0]), V.Survey(sex='M', age=40), backend)
    assert r['status'] == 'ok'
    r = {**r, 'hb_range': [104.0, 126.0], 'hb_note': 'в 80% случаев реальный Hb был в пределах ±11 г/л'}
    txt, _ = _pdf(tmp_path, r, V.Survey(sex='M', age=40), res=res)
    assert 'Ориентировочный Hb по фото: 104–126 г/л' in txt
    assert 'НЕ лабораторное измерение' in txt and '±11 г/л' in txt
    assert 'норма Hb ≥ 130 г/л' in txt and 'мужчины' in txt
    no_hb = {**r, 'hb_range': None, 'hb_note': 'цифра Hb не выдаётся: модель Hb почти не лучше константы'}
    txt, _ = _pdf(tmp_path, no_hb, V.Survey(sex='M', age=40), res=res)
    assert '104' not in txt and 'Hb по фото: не показан' in txt and 'не лучше константы' in txt


def test_partial_result_empty_fields(tmp_path, backend, monkeypatch):
    """Частичный: оценка есть, но нет фото, качества, анкеты, анализа и Personalizer — пустые поля без ошибок и
    без подставленных значений."""
    _route(monkeypatch, {'conj', 'sclera'})
    r, _ = V.run_analysis(synthetic_eye()[0], V.Survey(), backend)
    assert r['status'] == 'ok'
    r = {**r, 'crop': None, 'photo_preview': None}
    txt, rd = _pdf(tmp_path, r, V.Survey(), res=None, lab=None, photo=None, model_version='', reply=None)
    assert 'Пол не указан' in txt and 'Возраст не указан' in txt
    assert 'Изображения недоступны' in txt and _n_images(rd) == 0
    assert 'Проверка качества не выполнялась' in txt
    assert 'Ответов Personalizer нет' in txt and 'Не введён' in txt
    assert 'модель не указана' in txt
    assert 'пол не указан: применены нормы женщин' in txt                    # как who_limits(None, None)
    assert R.DISCLAIMER in txt


@pytest.mark.parametrize('case', ['age', 'dark'])
def test_refusal_has_no_level_and_no_numbers(tmp_path, backend, case):
    """Отказ ядра (ребёнок < 5 лет, тёмный снимок) → без уровня, без Hb, без норм; причина и дисклеймер есть."""
    img, survey = ((synthetic_eye()[0], V.Survey(sex='F', age=3)) if case == 'age' else
                   (synthetic_eye()[0] * 0.05, V.Survey(sex='M', age=40)))
    r, res = V.run_analysis(img, survey, backend)
    assert r['status'] != 'ok'
    stub = P.StubPersonalizer().reply(P.make_input(r, survey))
    txt, _ = _pdf(tmp_path, r, survey, res=res, photo=None, model_version=backend.version, reply=stub)
    assert 'РИСК НЕ ОПРЕДЕЛЁН' in txt and not any(w in txt for w in LEVEL_WORDS)
    assert 'г/л' not in txt and 'Hb по фото: не оценивался' in txt
    assert 'Не применялись' in txt
    assert 'Ответов Personalizer нет' in txt                                 # заглушка не печатается как ответ
    for issue in r['issues']:
        assert ' '.join(issue.split())[:25] in txt
    assert R.DISCLAIMER in txt


def test_refusal_ignores_stale_numbers(tmp_path):
    """Даже если в словаре отказа остались риск и Hb (unreliable/ood), в PDF их нет."""
    r = {'status': 'unreliable', 'stage': 'ood', 'risk': 'высокий', 'p_anemia': 0.9, 'hb_range': (80.0, 100.0),
         'hb_note': 'экспериментально', 'issues': ['снимок не похож на обучающие'], 'notes': [],
         'reasons': ['P(анемия) = 0.90'], 'model': 'm', 'crop': None}
    txt, _ = _pdf(tmp_path, r, V.Survey(sex='F', age=30))
    assert 'ВЫСОКИЙ РИСК' not in txt and '80–100' not in txt and 'P(анемия)' not in txt and 'г/л' not in txt


def test_cyrillic_fonts_embedded(tmp_path):
    """Кириллица — встроенным TTF (не Helvetica без кириллицы): текст извлекается обратно без потерь."""
    r = {'status': 'retake', 'stage': 'frame', 'issues': ['Ёлка: щёлочь, съешь же ещё этих мягких булок'],
         'notes': [], 'reasons': []}
    txt, rd = _pdf(tmp_path, r, V.Survey(sex='F', age=30, wellbeing={'weakness': None}))
    assert 'Ёлка: щёлочь, съешь же ещё этих мягких булок' in txt
    fonts = str(rd.pages[0]['/Resources']['/Font'])
    assert 'Helvetica' not in fonts


def test_font_not_found(monkeypatch):
    import sys
    monkeypatch.setattr(R, 'FONT_CANDIDATES', (('/nope/a.ttf', '/nope/b.ttf'),))
    monkeypatch.setenv('HEMAVISION_FONT', '/nope/c.ttf')
    monkeypatch.setitem(sys.modules, 'matplotlib', None)
    with pytest.raises(R.FontNotFound):
        R.find_fonts()


def test_references_match_core_norms():
    """Нормы в отчёте — из labs.who_limits (как в ядре), подпись группы согласована с ветвями who_limits."""
    for sex, age, preg in [('F', 30, False), ('F', 30, True), ('M', 40, None), (None, None, None), ('M', 13, None),
                           ('F', 8, None)]:
        s = V.Survey(sex=sex, age=age, pregnant=preg)
        (_, val, src), = R.references({'status': 'ok'}, s)
        assert f'≥ {LABS.who_limits(sex, age, preg).normal:.0f} г/л' in val and 'ВОЗ 2011' in src
    assert R.references({'status': 'retake'}, V.Survey(sex='F', age=30)) == []


def test_personalizer_entries():
    rep = P.PersonalizerReply(title='AI Personalizer', text='**Средний** риск', available=True, source='rules')
    hist = ({'role': 'user', 'text': 'x', 'intro': True}, {'role': 'assistant', 'text': 'y', 'intro': True},
            {'role': 'user', 'text': 'Можно ли есть гречку?'}, {'role': 'assistant', 'text': '**Да**, можно.'})
    assert R.personalizer_entries(rep, hist) == [('AI Personalizer', 'Средний риск'),
                                                 ('Вопрос', 'Можно ли есть гречку?'), ('Ответ', 'Да, можно.')]
    off = dataclasses.replace(rep, available=False)
    assert R.personalizer_entries(off, ()) == [] and R.personalizer_entries(None, None) == []


def test_ui_make_pdf_from_session_state(tmp_path, backend, monkeypatch):
    """Кнопка «Скачать PDF» интерфейса: состояние экрана результата + диалог → PDF (без gradio)."""
    from app.hemavision.ui import make_pdf
    _route(monkeypatch, {'conj', 'sclera'})
    photo = _jpeg(synthetic_eye()[0])
    survey = V.Survey(sex='F', age=30)
    r, res = V.run_analysis(photo, survey, backend)
    reply = P.RulesPersonalizer().reply(P.make_input(r, survey))
    st = {'r': r, 'survey': survey, 'lab': None, 'res': res, 'photo': photo, 'reply': reply}
    chat = {'history': P.start_history(reply) + ({'role': 'user', 'text': 'Что такое гемоглобин?'},
                                                  {'role': 'assistant', 'text': 'Белок крови.'})}
    path = make_pdf(st, chat, backend.version)
    assert path.endswith('.pdf')
    txt, rd = _text(path)
    assert 'Вопрос: Что такое гемоглобин?' in txt and 'Ответ: Белок крови.' in txt and backend.version in txt
    assert _n_images(rd) >= 2
