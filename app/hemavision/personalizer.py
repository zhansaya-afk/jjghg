"""AI Personalizer для интерфейса HemaVision: интерфейс, офлайн-адаптер движка правил, заглушка и диалог.

UI работает только через Personalizer.reply(PersonalizerInput) → PersonalizerReply, поэтому движок подключается
одним адаптером (класс с тем же методом) без правки экранов: build(personalizer=МойАдаптер()).
RulesPersonalizer — движок anemia.personalizer (правила + база знаний): работает офлайн, без сети и языковых моделей.

Правила для любого адаптера: ответ анкеты не меняет риск по фото; без диагнозов, названий препаратов и доз;
фото в адаптер не передаётся (только поля результата analyze_photo). Анализ крови — отдельный блок lab
(labs.lab_hb_report, как Я3): с прогнозом Hb по фото не смешивается и его не меняет.
Диалог (ask): вопросы и ответы этой сессии интерфейса передаются движку как history — так же, как в Я5.
Ключи самочувствия совпадают с опросом персонализатора: weakness, blood_loss, meat_fish (True / False / None).
"""
from dataclasses import dataclass, field, replace
from typing import Protocol, runtime_checkable

WELLBEING = (   # (ключ, вопрос) — ключи как в опросе персонализатора
    ('weakness', 'Бывают ли слабость, одышка или головокружение при обычной нагрузке?'),
    ('blood_loss', ('Есть ли кровопотери: обильные месячные, донорство чаще раза в 3 месяца, частые носовые '
                    'кровотечения?')),
    ('meat_fish', 'Едите ли вы мясо или рыбу хотя бы 2–3 раза в неделю?'),
)


@dataclass(frozen=True)
class PersonalizerInput:
    status: str                         # 'ok' | 'retake' | 'unreliable' — из analyze_photo
    risk: str | None                    # 'низкий' | 'средний' | 'высокий' | None (оценки нет)
    hb_range: tuple | None              # (от, до) г/л или None — None всегда, если оценки нет
    sex: str | None                     # 'M' | 'F' | None
    age: float | None
    pregnant: bool | None
    wellbeing: dict = field(default_factory=dict)   # {ключ WELLBEING: True / False / None}
    question: str | None = None         # вопрос пользователя (на будущее: чат)
    stage: str | None = None            # этап analyze_photo, на котором решён ответ ('ok', 'frame', 'age', 'ood', ...)
    p_anemia: float | None = None       # вероятность анемии по фото; None всегда, если оценки нет
    tissues: tuple = ()                 # ткани, по которым оценено: ('conj',), ('sclera',), ('conj', 'sclera')
    issues: tuple = ()                  # причины, почему снимок не подошёл (тексты analyze_photo)
    lab: dict | None = None             # готовый анализ крови: dict labs.lab_hb_report (как Я3) или None
    history: tuple = ()                 # прошлые реплики диалога: ({'role': 'user'|'assistant', 'text': ...}, ...)
    hb_estimate: float | None = None    # прогноз Hb по фото (г/л, analyze_photo); None — середина hb_range


@dataclass(frozen=True)
class PersonalizerReply:
    title: str
    text: str                           # Markdown
    available: bool                     # False — заглушка / модуль не подключён
    source: str                         # кто ответил: 'stub', 'rules', ...


@runtime_checkable
class Personalizer(Protocol):
    name: str

    def reply(self, inp: PersonalizerInput) -> PersonalizerReply: ...


class StubPersonalizer:
    """Заглушка: ничего не советует, только говорит, что модуль в разработке и что ответы анкеты сохранены."""
    name = 'stub'

    def reply(self, inp: PersonalizerInput) -> PersonalizerReply:
        answered = sum(v is not None for v in inp.wellbeing.values())
        lines = [('Модуль **AI Personalizer** в разработке и подключится в следующей версии. Он будет объяснять '
                  'результат простыми словами с учётом анкеты.')]
        if answered:
            lines.append(f'Ответов о самочувствии: {answered} из {len(WELLBEING)}. Они записаны в отчёт для врача '
                         'и **на оценку риска по фото не влияют**.')
        else:
            lines.append('Вопросы о самочувствии пропущены: на оценку риска по фото они не влияют.')
        return PersonalizerReply(title='AI Personalizer · скоро', text='\n\n'.join(lines), available=False,
                                 source=self.name)


RULES_TITLES = {'urgent': 'AI Personalizer · срочно к врачу', 'child': 'AI Personalizer · детям до 5 лет',
                'retake': 'AI Personalizer · переснимите фото'}


class RulesPersonalizer:
    """Офлайн-движок персонализатора (anemia.personalizer): правила + проверенная база знаний, без сети.
    Только объясняет результат ядра: риск, Hb и ткань берутся из analyze_photo и не пересчитываются."""
    name = 'rules'

    def reply(self, inp: PersonalizerInput) -> PersonalizerReply:
        from anemia.personalizer import personalize

        result = {'status': inp.status, 'stage': inp.stage, 'risk': inp.risk, 'p_anemia': inp.p_anemia,
                  'hb_range': list(inp.hb_range) if inp.hb_range else None, 'hb_estimate': inp.hb_estimate,
                  'tissues': list(inp.tissues),
                  'issues': list(inp.issues)}
        profile = {'sex': inp.sex, 'age': inp.age, 'pregnant': inp.pregnant}
        survey = dict(inp.wellbeing) if inp.wellbeing else None
        try:
            out = personalize(result, inp.lab, profile, survey, question=inp.question or None,
                              history=list(inp.history) or None)
        except Exception:   # noqa: BLE001 — ошибка движка не должна ломать экран результата
            return PersonalizerReply(title='AI Personalizer', available=False, source=self.name,
                                     text=('Объяснение сейчас недоступно. Результат по фото выше не изменился; '
                                           'окончательный ответ дают анализ крови и врач.'))
        title = RULES_TITLES.get(out['mode'], 'AI Personalizer')
        return PersonalizerReply(title=title, text=out['text'], available=True, source=self.name)


def make_input(r, survey, lab=None):
    """Результат analyze_photo + анкета (view.Survey) [+ анализ крови lab_hb_report] → PersonalizerInput.
    Без оценки по фото — без риска, P и Hb по фото (анализ крови при этом остаётся: это отдельный путь).
    Фото и картинки (crop, маски) не передаются: только поля результата."""
    ok = r.get('status') == 'ok'
    p = r.get('p_anemia')
    return PersonalizerInput(status=r.get('status', 'retake'), risk=r.get('risk') if ok else None,
                             hb_range=tuple(r['hb_range']) if ok and r.get('hb_range') else None,
                             sex=survey.sex, age=survey.age, pregnant=survey.pregnant,
                             wellbeing=dict(survey.wellbeing), stage=r.get('stage'),
                             p_anemia=float(p) if ok and p is not None else None,
                             tissues=tuple(r.get('tissues') or ()) if ok else (),
                             issues=tuple(str(x) for x in r.get('issues') or ()), lab=lab,
                             hb_estimate=r.get('hb_estimate') if ok and r.get('hb_range') else None)


# ------------------------------------------------------------------ диалог (как ячейка Я5, но в интерфейсе)
MAX_TURNS = 30                          # вопросов за одну проверку (как session.chat в Я5)
MAX_QUESTION = 500                      # символов в одном вопросе: длиннее — обрезается
LIMIT_TEXT = ('Лимит вопросов для этой проверки исчерпан. Начните новую проверку или обсудите вопросы с врачом: '
              'окончательный ответ дают анализ крови и врач.')


def start_history(reply):
    """История диалога после первого объяснения (как в Я5: «Что значит мой результат?» → объяснение).
    Объяснение недоступно (заглушка, ошибка) → пустая история."""
    from anemia.personalizer.session import TEXT
    if not reply.available:
        return ()
    return ({'role': 'user', 'text': TEXT['ru']['explain'], 'intro': True},
            {'role': 'assistant', 'text': reply.text, 'intro': True})


def ask(personalizer, inp, history, question):
    """Один вопрос в диалоге → (PersonalizerReply | None, новая история). Пустой вопрос → (None, та же история).
    Контекст — только история этой проверки; риск, Hb и анализ крови из inp не меняются. Язык ответа — язык
    вопроса (русский или казахский), правила безопасности (срочно, дети, дозы, диагноз) — движка."""
    q = ' '.join(str(question or '').split())[:MAX_QUESTION]
    history = tuple(history or ())
    if not q:
        return None, history
    if sum(m['role'] == 'user' and not m.get('intro') for m in history) >= MAX_TURNS:
        return PersonalizerReply(title='AI Personalizer', text=LIMIT_TEXT, available=False,
                                 source=getattr(personalizer, 'name', '?')), history
    rep = personalizer.reply(replace(inp, question=q, history=history))
    if not rep.available:               # ошибка или заглушка: в историю не записываем
        return rep, history
    return rep, history + ({'role': 'user', 'text': q}, {'role': 'assistant', 'text': rep.text})


def chat_messages(history):
    """История → сообщения gr.Chatbot. Первое объяснение (intro) показано карточкой выше, в чате только вопросы."""
    return [{'role': m['role'], 'content': m['text']} for m in history if not m.get('intro')]
