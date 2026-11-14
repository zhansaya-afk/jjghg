"""Проверка базы знаний персонализатора (то же, что ячейка Я6 в Colab).

python scripts/check_kb.py                                     # формат, запреты, копии экзамена, экзамен
python scripts/check_kb.py --questions "анализ после еды | можно ли кофе"   # какую тему находит вопрос
python scripts/check_kb.py --kb черновик.md --no-exam          # проверить другой файл (без экзамена)
Код выхода 1 — есть ошибки. Как дополнять базу: anemia/personalizer/kb/README.md.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from anemia.personalizer.kbtools import check_kb


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--kb", help="файл базы (по умолчанию anemia/personalizer/kb/ru.md)"
    )
    ap.add_argument("--questions", default="", help="свои вопросы через |")
    ap.add_argument("--no-exam", action="store_true", help="не запускать экзамен")
    a = ap.parse_args(argv)
    return 0 if check_kb(a.kb, a.questions, exam=not a.no_exam) else 1


if __name__ == "__main__":
    sys.exit(main())
