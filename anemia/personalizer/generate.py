"""Необязательная локальная LLM. Две роли, обе под контролем режима rules:
- pick_topic — понять перефразированный вопрос, который поиск по буквам не нашёл, и выбрать тему базы
  (или «нет темы»); текст ответа остаётся проверенным текстом базы;
- rewrite — переформулировать готовый ответ (страж проверяет; при нарушении остаётся ответ rules).

Модель: Qwen2.5-3B-Instruct GGUF Q4_K_M через llama-cpp-python (pip install llama-cpp-python).
Скачивается в Colab с Hugging Face; из облачной сессии разработки HF недоступен, поэтому этот модуль
проверен только на поддельной модели в тестах. Реальная скорость и качество — по экзамену в Colab.
"""

import json
import re

REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
SYSTEM = (
    "Ты помощник приложения скрининга анемии. Перепиши черновик ответа простыми словами для человека без "
    "медицинского образования. Правила: используй только факты из черновика и карточки; не добавляй цифр; "
    "не ставь диагноз; не называй лекарства и дозы; сохрани все цифры, уровень риска и совет обратиться к врачу; "
    "отвечай на языке вопроса; не длиннее 5 предложений."
)


def load_llm(
    path=None,
    n_ctx=6144,
    n_gpu_layers=-1,
    max_tokens=220,
    temperature=0.2,
    n_threads=None,
):
    """→ callable(messages) → str. path=None — скачать REPO/FILE с Hugging Face (нужен интернет в Colab)."""
    import os

    from llama_cpp import Llama

    # по умолчанию llama.cpp берёт половину ядер; в Colab их 2 — нужны оба
    opts = {
        "n_ctx": n_ctx,
        "n_gpu_layers": n_gpu_layers,
        "n_threads": n_threads or os.cpu_count(),
        "verbose": False,
    }
    if path:
        model = Llama(model_path=path, **opts)
    else:
        model = Llama.from_pretrained(REPO, FILE, **opts)

    def chat(messages, max_tokens=max_tokens, temperature=temperature):
        # одинаковое начало сообщений (список тем) llama.cpp не пересчитывает: считается только вопрос
        r = model.create_chat_completion(
            messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        return r["choices"][0]["message"]["content"]

    return chat


def build_messages(question, facts, draft, history=None):
    card = {k: v for k, v in facts.items() if v}
    msgs = [{"role": "system", "content": SYSTEM}]
    for h in (history or [])[-4:]:
        msgs.append({"role": h["role"], "content": h["text"]})
    msgs.append(
        {
            "role": "user",
            "content": f"Вопрос: {question or 'Объясни мой результат'}\n"
            f"Карточка фактов: {json.dumps(card, ensure_ascii=False)}\n"
            f"Черновик ответа: {draft['text']}\n"
            "Перепиши черновик по правилам.",
        }
    )
    return msgs


def rewrite(llm, question, facts, draft, history=None):
    try:
        return (llm(build_messages(question, facts, draft, history)) or "").strip()
    except Exception:  # noqa: BLE001 — любая ошибка модели → остаётся ответ rules
        return ""


TOPIC_SYSTEM = (
    "Ты определяешь тему вопроса в приложении скрининга анемии. Ниже список тем: id и примеры вопросов. "
    "Выбери одну тему, которая по смыслу отвечает на вопрос пользователя, даже если слова другие. "
    "Ответь только id темы, без пояснений. Если ни одна тема не отвечает на этот вопрос — ответь NONE."
)


def topic_catalog(fragments, per_topic=4):
    """Список тем для подсказки модели: «id: пример | пример | …» (одинаковый для всех вопросов).
    Примеры берутся вразброс по списку формулировок темы: первые обычно почти одинаковые."""
    lines = []
    for fr in fragments:
        ex = fr.examples
        if not ex:
            continue
        k = min(per_topic, len(ex))
        idx = sorted({round(i * (len(ex) - 1) / max(k - 1, 1)) for i in range(k)})
        lines.append(f"{fr.id}: {' | '.join(ex[i] for i in idx)}")
    return "\n".join(lines)


def topic_messages(question, fragments):
    return [
        {
            "role": "system",
            "content": TOPIC_SYSTEM + "\n\nТемы:\n" + topic_catalog(fragments),
        },
        {"role": "user", "content": f"Вопрос: {question}\nid темы:"},
    ]


def parse_topic(reply, ids):
    """Ответ модели → id темы или None. Берётся первое слово-id; NONE, пусто, чужое слово → None."""
    for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", reply or ""):
        if tok.upper() == "NONE":
            return None
        if tok in ids:
            return tok
    return None


def pick_topic(llm, question, fragments):
    """Тема базы для вопроса, который поиск не нашёл, или None. Ошибка модели → None (остаётся «не знаю»)."""
    msgs = topic_messages(question, fragments)
    try:
        try:
            reply = llm(msgs, max_tokens=16, temperature=0.0)
        except (
            TypeError
        ):  # простая llm(messages) без параметров (поддельная модель в тестах)
            reply = llm(msgs)
    except Exception:  # noqa: BLE001
        return None
    return parse_topic(reply, {fr.id for fr in fragments})
