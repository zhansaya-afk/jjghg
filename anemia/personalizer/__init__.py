"""AI-персонализатор: объясняет результат скрининга и анализа крови, отвечает на вопросы по базе знаний.

Не ставит диагноз, не назначает лечение, не меняет риск по фото. Работает офлайн.
Проект: https://claude.ai/artifact/6aP9vPG37JFvsqFYDEExKZ
"""

from .engine import answer, personalize, rules_answer
from .facts import MODEL_CARD, build_facts
from .survey import QUESTIONS, questions_for

__all__ = [
    "MODEL_CARD",
    "QUESTIONS",
    "answer",
    "build_facts",
    "personalize",
    "questions_for",
    "rules_answer",
]
