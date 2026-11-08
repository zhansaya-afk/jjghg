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


# слова, которые есть почти в каждом вопросе об анемии: по ним тема не различается, а совпадение
# «может … понизить» тянуло вопрос о донорстве к теме про стресс. Убираются только из поиска.
# «гемоглобин» оставлен: без него «какой у меня гемоглобин по фото?» терял тему hb_by_photo.
_COMMON_WORDS = """
    анемия анемии анемией анемию при низкий низком низкого может могут ли бывает правда вообще стоит
    влияет влиять малокровие малокровия малокровием малокровии
"""
COMMON = frozenset(_COMMON_WORDS.split())
_ASK_WORDS = """
    почему зачем сколько когда где какой какая какое какие каким чем куда откуда насколько как такое
"""
ASK = frozenset(_ASK_WORDS.split())
TOPIC_WEIGHT = 0.5  # доля оценки по теме целиком (все её формулировки вместе) против лучшей одной формулировки


def _key(s):
    """Текст для поиска: _norm без общих слов или None, если без них остались одни вопросительные слова
    («анемия?», «сколько было с анемией»): тогда ищется полный текст (_norm) по полным формулировкам."""
    rest = [w for w in _norm(s).split() if w not in COMMON]
    return " ".join(rest) if any(w not in ASK for w in rest) else None


@lru_cache(maxsize=4)
def load_kb(lang="ru"):
    return tuple(parse_kb((KB_DIR / f"{lang}.md").read_text(encoding="utf-8")))


class _Index:
    """TF-IDF (символьные n-граммы) по формулировкам и по темам целиком (все формулировки одним текстом)."""

    def __init__(self, per_fragment):
        from sklearn.feature_extraction.text import TfidfVectorizer

        def tfidf(texts):
            vec = TfidfVectorizer(
                analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True
            )
            return vec, vec.fit_transform(texts)

        self.vec, self.mat = tfidf([t for ts in per_fragment for t in ts])
        self.tvec, self.tmat = tfidf([" ".join(ts) for ts in per_fragment])

    def sims(self, q):
        """→ (сходство с каждой формулировкой, сходство с каждой темой)."""
        ex = (self.mat @ self.vec.transform([q]).T).toarray().ravel()
        return ex, (self.tmat @ self.tvec.transform([q]).T).toarray().ravel()


class Retriever:
    """Сходство вопроса с формулировками каждой темы. Оценка темы (TF-IDF) = половина — лучшая одна
    формулировка, половина — тема целиком (все формулировки одним текстом): так вопрос, похожий на
    несколько формулировок темы понемногу, не проигрывает случайному совпадению с одной чужой.
    Поиск идёт без общих слов (_key); вопрос из одних общих и вопросительных слов — по полному тексту.
    Всё это и порог engine.THRESHOLD = 0,35 — для TF-IDF; с encoder= (эмбеддер) поиск по полному тексту
    и лучшей формулировке, порог для него надо подбирать заново."""

    def __init__(self, fragments, encoder=None):
        self.fragments = list(fragments)
        self.owner = [i for i, fr in enumerate(self.fragments) for _ in fr.examples]
        self.encoder = encoder
        if encoder is None:
            full = [[_norm(e) for e in fr.examples] for fr in self.fragments]
            keyed = [
                [_key(e) or n for e, n in zip(fr.examples, ns)]
                for fr, ns in zip(self.fragments, full)
            ]
            self.keyed, self.full = _Index(keyed), _Index(full)
        else:
            self.mat = np.asarray(
                encoder([_norm(e) for fr in self.fragments for e in fr.examples]), float
            )
            self.mat /= np.linalg.norm(self.mat, axis=1, keepdims=True) + 1e-12

    def search(self, question, k=3):
        """→ [(Fragment, сходство 0–1)] по убыванию."""
        key, full = _key(question), _norm(question)
        if not full:
            return []
        topic = None
        if self.encoder is None:
            sims, topic = self.keyed.sims(key) if key else self.full.sims(full)
        else:
            v = np.asarray(self.encoder([full]), float).ravel()
            sims = self.mat @ (v / (np.linalg.norm(v) + 1e-12))
        best = {}
        for j, s in enumerate(sims):
            i = self.owner[j]
            best[i] = max(best.get(i, 0.0), float(s))
        if topic is not None:
            best = {
                i: (1 - TOPIC_WEIGHT) * b + TOPIC_WEIGHT * float(topic[i])
                for i, b in best.items()
            }
        order = sorted(best, key=best.get, reverse=True)[:k]
        return [(self.fragments[i], best[i]) for i in order]


@lru_cache(maxsize=4)
def default_retriever(lang="ru"):
    return Retriever(load_kb(lang))
