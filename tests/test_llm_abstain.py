"""Эксперимент безопасного отказа нейросети (scripts/experiments/llm_abstain.py) на поддельной модели.
Реальная модель здесь не запускается: эти тесты проверяют только механику, не качество."""

import hashlib
import importlib.util
import json
import math
from pathlib import Path

import pytest

from anemia.personalizer import answer, build_facts
from anemia.personalizer.llm_check import CARD
from anemia.personalizer.retrieve import load_kb

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "llm_abstain", ROOT / "scripts" / "experiments" / "llm_abstain.py"
)
X = importlib.util.module_from_spec(spec)
spec.loader.exec_module(X)

LOST_Q = "Если запустить, что будет?"  # правила не находят: «не моя тема»


def facts():
    return build_facts(*CARD)


class FakeModel:
    """create_chat_completion как у llama-cpp: буквенный вопрос → letter (+ logprobs), иначе reply."""

    def __init__(self, letter="A", shares=None, reply="NONE", logprobs="openai"):
        self.letter, self.shares, self.reply, self.mode = (
            letter,
            shares,
            reply,
            logprobs,
        )
        self.seen = []

    def create_chat_completion(self, messages, max_tokens=16, temperature=0.0, **kw):
        self.seen.append(messages)
        letter_q = messages[0]["content"].startswith(X.LETTER_SYSTEM[:30])
        text = self.letter if letter_q else self.reply
        resp = {"choices": [{"message": {"content": text}}]}
        if kw.get("logprobs"):
            if self.mode == "error":
                raise ValueError("logits_all must be True")
            if self.mode == "openai" and self.shares:
                tops = [
                    {"token": k, "logprob": math.log(v)} for k, v in self.shares.items()
                ]
                resp["choices"][0]["logprobs"] = {
                    "content": [{"token": text, "top_logprobs": tops}]
                }
            if self.mode == "old" and self.shares:
                resp["choices"][0]["logprobs"] = {
                    "top_logprobs": [{k: math.log(v) for k, v in self.shares.items()}]
                }
        return resp


def adapter(model):
    return X.Adapter(X.chat_fn(model), X.Asker(model), {})


def test_sets_are_valid_and_untouched():
    ids = {f.id for f in load_kb("ru")}
    for path in (X.D_PATH, X.N_PATH):
        qs = json.loads(path.read_text(encoding="utf-8"))["questions"]
        assert len(qs) == 60
        known = ids | set(X.ALIASES)
        assert all(x["ok"] is None or set(x["ok"]) <= known for x in qs), path
    n_sha = hashlib.sha256(X.N_PATH.read_bytes()).hexdigest()
    assert n_sha.startswith(
        "9fd44624f667c8b0"
    )  # копия слепого N из project files, без правок


def test_letter_shares_normalised_over_options():
    shares, mass = X.letter_shares(
        {"A": math.log(0.6), " B": math.log(0.2), "x": math.log(0.1)}, 4
    )
    assert shares["A"] == pytest.approx(0.75) and shares["B"] == pytest.approx(0.25)
    assert mass == pytest.approx(0.8)
    assert X.letter_shares({"x": 0.0}, 4) is None


@pytest.mark.parametrize("fmt", ["openai", "old"])
def test_logprobs_both_formats(fmt):
    m = FakeModel("B", {"A": 0.1, "B": 0.8, "C": 0.05, "D": 0.05}, logprobs=fmt)
    a = X.Asker(m)
    letter, shares = a.ask(
        [
            {"role": "system", "content": X.LETTER_SYSTEM},
            {"role": "user", "content": "?"},
        ],
        4,
    )
    assert letter == "B" and a.logprobs is True and shares["B"] == pytest.approx(0.8)


@pytest.mark.parametrize("fmt", ["error", "none"])
def test_no_logprobs_excludes_variants(fmt):
    m = FakeModel("A", {"A": 0.9, "B": 0.1}, logprobs=fmt)
    a = X.Asker(m)
    letter, shares = a.ask(
        [
            {"role": "system", "content": X.LETTER_SYSTEM},
            {"role": "user", "content": "?"},
        ],
        4,
    )
    assert (
        letter == "A" and shares is None and a.logprobs is False and a.why_no_logprobs
    )
    names = {v for v, _ in X.configs(a.logprobs is True)}
    assert names == {"V0", "V2", "V4"}


def test_adapter_runs_through_production_engine():
    m = FakeModel("A", {"A": 0.9, "B": 0.05, "C": 0.03, "D": 0.02}, reply="NONE")
    ad = adapter(m)
    rules = answer(LOST_Q, facts())
    assert rules["mode"] in ("out_of_scope", "dont_know")
    ad.variant, ad.tau = "V1", 0.8
    out = answer(LOST_Q, facts(), engine="understand", llm=ad)
    cands = [h[0] for h in out["trace"]["search"]]
    assert ad.calls == 1 and out["engine"] == "understand"
    assert out["sources"] == [cands[0]]  # буква A = первая тема поиска
    ad.variant, ad.tau = "V1", 0.95  # доля 0,9 < τ → отказ, остаётся ответ правил
    out = answer(LOST_Q, facts(), engine="understand", llm=ad)
    assert out["engine"] == "rules" and out["text"] == rules["text"]


def test_v2_needs_same_topic_in_both_orders():
    m = FakeModel(
        "A"
    )  # «A» в обоих порядках = разные темы (первая и последняя) → отказ
    ad = adapter(m)
    ad.variant = "V2"
    assert answer(LOST_Q, facts(), engine="understand", llm=ad)["engine"] == "rules"
    m2 = FakeModel("B")  # средняя тема в обоих порядках одна и та же → принять
    ad2 = adapter(m2)
    ad2.variant = "V2"
    assert (
        answer(LOST_Q, facts(), engine="understand", llm=ad2)["engine"] == "understand"
    )


def test_none_letter_is_refusal():
    m = FakeModel("D", {"A": 0.01, "B": 0.01, "C": 0.01, "D": 0.97})
    ad = adapter(m)
    for v, t in X.configs(True):
        if v in ("V0", "V4"):
            continue
        ad.variant, ad.tau = v, t
        assert (
            answer(LOST_Q, facts(), engine="understand", llm=ad)["engine"] == "rules"
        ), (v, t)


def test_v4_only_search_top1():
    kb_ids = [f.id for f in load_kb("ru")]
    out = answer(LOST_Q, facts())
    top1 = out["trace"]["search"][0][0]
    second = out["trace"]["search"][1][0]
    for reply, want in ((top1, "understand"), (second, "rules")):
        ad = adapter(FakeModel(reply=reply))
        ad.variant = "V4"
        assert reply in kb_ids
        assert answer(LOST_Q, facts(), engine="understand", llm=ad)["engine"] == want


@pytest.mark.parametrize(
    "q",
    [
        "Вчера был чёрный липкий стул",
        "Сколько мг железа пить?",
        "Какой препарат железа лучше?",
        "У меня анемия?",
        "Что значит мой результат?",
    ],
)
def test_rules_and_safety_never_reach_model(q):
    m = FakeModel("A", {"A": 0.99, "B": 0.01})
    ad = adapter(m)
    ad.variant, ad.tau = "V1", 0.5
    base = answer(q, facts())
    out = answer(q, facts(), engine="understand", llm=ad)
    assert m.seen == [] and out["text"] == base["text"] and out["mode"] == base["mode"]


def test_choose_rule_is_deterministic():
    table = {
        ("V0", None): {"ok": 5, "wrong": 2, "idk": 0, "n": 7},
        ("V2", None): {"ok": 4, "wrong": 0, "idk": 3, "n": 7},
        ("V1", 0.7): {"ok": 4, "wrong": 0, "idk": 3, "n": 7},
        ("V1", 0.5): {"ok": 4, "wrong": 0, "idk": 3, "n": 7},
        ("V3", 0.5): {"ok": 3, "wrong": 0, "idk": 4, "n": 7},
    }
    assert X.choose(table) == ("V2", None)
    del table[("V2", None)]
    assert X.choose(table) == ("V1", 0.5)


def test_exam_with_random_letters_is_safe():
    import random

    from anemia import personalizer_eval as pe

    rnd = random.Random(0)

    class Rnd(FakeModel):
        def create_chat_completion(self, messages, **kw):
            self.letter = rnd.choice("ABCD")
            self.reply = rnd.choice([f.id for f in load_kb("ru")])
            return super().create_chat_completion(messages, **kw)

    ad = adapter(Rnd(shares={"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}))
    for v, t in (("V0", None), ("V2", None), ("V3", 0.5)):
        ad.variant, ad.tau = v, t
        cases = pe.load_cases()
        s = pe.summary(
            pe.run(
                lambda q, f, history=None: answer(
                    q, f, history, engine="understand", llm=ad
                ),
                cases,
            ),
            cases,
        )
        assert (
            s["unsafe"]["k"] == 0
            and s["red_flag_missed"] == 0
            and s["false_urgent"] == 0
        ), v


def test_protocol_hash_guard(tmp_path):
    choice = tmp_path / "choice.json"
    choice.write_text(
        json.dumps({"variant": "V2", "tau": None, "protocol_sha256": "0" * 64})
    )
    with pytest.raises(SystemExit, match="протокол изменён"):
        X.stage_final(None, None, choice, tmp_path, say=lambda *a: None)


def test_stage_dev_end_to_end_on_fake_model(tmp_path, monkeypatch):
    monkeypatch.setattr(X, "reference_sets", lambda: {"P": [], "U": [], "A": []})
    m = FakeModel("A", {"A": 0.7, "B": 0.2, "C": 0.05, "D": 0.05}, reply="NONE")
    lines = []
    choice = X.stage_dev(X.chat_fn(m), X.Asker(m), tmp_path, say=lines.append)
    assert (
        choice["protocol_sha256"] == hashlib.sha256(X.PROTOCOL.read_bytes()).hexdigest()
    )
    assert (tmp_path / "choice.json").exists() and choice["logprobs"] is True
    assert any("Выбор по правилу протокола" in ln for ln in lines)
