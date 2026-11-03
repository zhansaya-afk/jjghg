"""Необязательная локальная LLM: только переформулирует готовый ответ режима rules.

Модель: Qwen2.5-3B-Instruct GGUF Q4_K_M через llama-cpp-python (pip install llama-cpp-python).
Скачивается в Colab с Hugging Face; из облачной сессии разработки HF недоступен, поэтому этот модуль
проверен только на поддельной модели в тестах. Реальная скорость и качество — по экзамену в Colab.
"""

import json

REPO = "Qwen/Qwen2.5-3B-Instruct-GGUF"
FILE = "qwen2.5-3b-instruct-q4_k_m.gguf"
SYSTEM = (
    "Ты помощник приложения скрининга анемии. Перепиши черновик ответа простыми словами для человека без "
    "медицинского образования. Правила: используй только факты из черновика и карточки; не добавляй цифр; "
    "не ставь диагноз; не называй лекарства и дозы; сохрани все цифры, уровень риска и совет обратиться к врачу; "
    "отвечай на языке вопроса; не длиннее 5 предложений."
)


def load_llm(path=None, n_ctx=2048, n_gpu_layers=-1, max_tokens=220, temperature=0.2):
    """→ callable(messages) → str. path=None — скачать REPO/FILE с Hugging Face (нужен интернет в Colab)."""
    from llama_cpp import Llama

    if path:
        model = Llama(
            model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False
        )
    else:
        model = Llama.from_pretrained(
            REPO, FILE, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False
        )

    def chat(messages):
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
