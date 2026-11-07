"""AI Personalizer: интерфейс для интерфейса HemaVision + временная заглушка.

Движок персонализатора (правила + база знаний) пока в отдельной ветке и в интерфейс не подключён. UI работает
только через Personalizer.reply(PersonalizerInput) → PersonalizerReply, поэтому движок подключается одним
адаптером (класс с тем же методом) без правки экранов: build(personalizer=МойАдаптер()).

Правила для любого адаптера: ответ анкеты не меняет риск по фото; без диагнозов, названий препаратов и доз.
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


def make_input(r, survey):
    """Результат analyze_photo + анкета (view.Survey) → PersonalizerInput. Без оценки — без риска и Hb."""
    ok = r.get('status') == 'ok'
    return PersonalizerInput(status=r.get('status', 'retake'), risk=r.get('risk') if ok else None,
                             hb_range=tuple(r['hb_range']) if ok and r.get('hb_range') else None,
                             sex=survey.sex, age=survey.age, pregnant=survey.pregnant,
                             wellbeing=dict(survey.wellbeing))
