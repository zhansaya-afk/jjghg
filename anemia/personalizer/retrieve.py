"""База знаний (kb/*.md) и поиск по ней.

По умолчанию — TF-IDF по символьным n-граммам (sklearn, без скачиваний): устойчив к окончаниям и опечаткам
(«анимия», «рез-т»). Эмбеддер multilingual-e5 можно подключить позже в Colab через encoder=.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

KB_DIR = Path(__file__).resolve().parent / "kb"
SLOT = re.compile(r"\{(\w+)\}")


@dataclass
class Fragment:
    id: str
    examples: list
    body: str
    src: str = ""
    mode: str = "answer"
    slots: list = field(default_factory=list)


def parse_kb(text):
    """Markdown базы знаний → список Fragment (заголовки «## id», строки q:/src:/mode:, тело после ---)."""
    out = []
    for block in re.split(r"^## ", text, flags=re.MULTILINE)[1:]:
        head, _, body = block.partition("\n---\n")
        lines = head.strip().splitlines()
        meta = {
            k.strip(): v.strip()
            for k, _, v in (ln.partition(":") for ln in lines[1:])
            if v
        }
        body = " ".join(body.split())
        out.append(
            Fragment(
                id=lines[0].strip(),
                examples=[e.strip() for e in meta.get("q", "").split("|") if e.strip()],
                body=body,
                src=meta.get("src", ""),
                mode=meta.get("mode", "answer"),
                slots=SLOT.findall(body),
            )
        )
    return out


# служебные слова: без них символьные n-граммы тянут вопрос к фрагменту по «можно ли …», а не по смыслу
_STOP_WORDS = """
    что как это мой моя мои мое моё у меня мне я а и в во на по же ли ну но или то так там тут уже был была было бы
    вы вас вам ваш он она они его ее их с со к ко о об от до за из для при про если когда чтобы можно нужно надо
    скажи скажешь расскажи еще очень просто вообще все всё весь ведь ты тебе не да себя сам сама есть
"""
STOP = frozenset(_STOP_WORDS.split())


_EAT = re.compile(
    r"\b(что|чего|можно|нельзя|нужно|надо|стоит|лучше|больше)\s+(?:мне\s+|нам\s+)?есть\b"
)


def _norm(s):
    s = re.sub(r"[^\w\s/%]", " ", (s or "").lower().replace("ё", "е"))
    s = _EAT.sub(
        r"\1 еда", s
    )  # «что мне есть» — про еду; иначе «есть» служебное («у меня есть анализ»)
    return " ".join(w for w in s.split() if w not in STOP)


@lru_cache(maxsize=4)
def load_kb(lang="ru"):
    return tuple(parse_kb((KB_DIR / f"{lang}.md").read_text(encoding="utf-8")))


class Retriever:
    """Сходство вопроса с примерами вопросов каждого фрагмента; оценка фрагмента = лучший пример."""

    def __init__(self, fragments, encoder=None):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.fragments = list(fragments)
        self.owner = [i for i, fr in enumerate(self.fragments) for _ in fr.examples]
        texts = [_norm(e) for fr in self.fragments for e in fr.examples]
        self.encoder = encoder
        if encoder is None:
            self.vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True
            )
            self.mat = self.vec.fit_transform(texts)
        else:
            self.mat = np.asarray(encoder(texts), float)
            self.mat /= np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12

    def search(self, question, k=3):
        """→ [(Fragment, сходство 0–1)] по убыванию."""
        q = _norm(question)
        if not q:
            return []
        if self.encoder is None:
            sims = (self.mat @ self.vec.transform([q]).T).toarray().ravel()
        else:
            v = np.asarray(self.encoder([q]), float).ravel()
            sims = self.mat @ (v / (np.linalg.norm(v) + 1e-12))
        best = {}
        for j, s in enumerate(sims):
            i = self.owner[j]
            best[i] = max(best.get(i, 0.0), float(s))
        order = sorted(best, key=best.get, reverse=True)[:k]
        return [(self.fragments[i], best[i]) for i in order]


@lru_cache(maxsize=4)
def default_retriever(lang="ru"):
    return Retriever(load_kb(lang))
