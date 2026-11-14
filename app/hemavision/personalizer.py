"""AI Personalizer для интерфейса HemaVision: интерфейс, офлайн-адаптер движка правил и заглушка.

UI работает только через Personalizer.reply(PersonalizerInput) → PersonalizerReply, поэтому движок подключается
одним адаптером (класс с тем же методом) без правки экранов: build(personalizer=МойАдаптер()).
RulesPersonalizer — движок anemia.personalizer (правила + база знаний): работает офлайн, без сети и языковых моделей.

Правила для любого адаптера: ответ анкеты не меняет риск по фото; без диагнозов, названий препаратов и доз;
фото в адаптер не передаётся (только поля результата analyze_photo).
Ключи самочувствия совпадают с опросом персонализатора: weakness, blood_loss, meat_fish (True / False / None).
"""
from dataclasses import dataclass, field
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
                  'hb_range': list(inp.hb_range) if inp.hb_range else None, 'tissues': list(inp.tissues),
                  'issues': list(inp.issues)}
        profile = {'sex': inp.sex, 'age': inp.age, 'pregnant': inp.pregnant}
        survey = dict(inp.wellbeing) if inp.wellbeing else None
        try:
            out = personalize(result, None, profile, survey, question=inp.question or None)
        except Exception:   # noqa: BLE001 — ошибка движка не должна ломать экран результата
            return PersonalizerReply(title='AI Personalizer', available=False, source=self.name,
                                     text=('Объяснение сейчас недоступно. Результат по фото выше не изменился; '
                                           'окончательный ответ дают анализ крови и врач.'))
        title = RULES_TITLES.get(out['mode'], 'AI Personalizer')
        return PersonalizerReply(title=title, text=out['text'], available=True, source=self.name)


def make_input(r, survey):
    """Результат analyze_photo + анкета (view.Survey) → PersonalizerInput. Без оценки — без риска, P и Hb.
    Фото и картинки (crop, маски) не передаются: только поля результата."""
    ok = r.get('status') == 'ok'
    p = r.get('p_anemia')
    return PersonalizerInput(status=r.get('status', 'retake'), risk=r.get('risk') if ok else None,
                             hb_range=tuple(r['hb_range']) if ok and r.get('hb_range') else None,
                             sex=survey.sex, age=survey.age, pregnant=survey.pregnant,
                             wellbeing=dict(survey.wellbeing), stage=r.get('stage'),
                             p_anemia=float(p) if ok and p is not None else None,
                             tissues=tuple(r.get('tissues') or ()) if ok else (),
                             issues=tuple(str(x) for x in r.get('issues') or ()))
