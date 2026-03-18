import os

import ollama

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "Answer in the same language as the user's question. "
    "Be concise, clear, and practical."
)
GENERAL_MODEL_NAME = os.getenv("GENERAL_LLM_MODEL", "tinyllama")


def ask_general_llm(question: str) -> str:
    if not question or not question.strip():
        return "Please provide a question."

    prompt = (
        f"{SYSTEM_PROMPT}\n"
        "Answer directly. Do not add unrelated examples.\n\n"
        f"Question: {question.strip()}\n"
        "Answer:"
    )

    try:
        response = ollama.generate(
            model=GENERAL_MODEL_NAME,
            prompt=prompt,
            options={"temperature": 0.0, "num_predict": 256},
        )
        content = response.get("response", "").strip()
        if not content:
            return "No response returned from the assistant model."
        return content
    except Exception as exc:
        return (
            "General assistant is unavailable right now. "
            f"Please ensure Ollama is running and model '{GENERAL_MODEL_NAME}' is available. Error: {exc}"
        )
