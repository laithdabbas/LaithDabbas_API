import os

import ollama

SYSTEM_PROMPT = (
    "You are a helpful, friendly assistant that helps users with general questions. "
    "Be concise, clear, and practical."
)
GENERAL_MODEL_NAME = os.getenv("GENERAL_LLM_MODEL", "llama3")


def ask_general_llm(question: str) -> str:
    if not question or not question.strip():
        return "Please provide a question."

    user_prompt = f"""
    {SYSTEM_PROMPT}

    Question: {question.strip()}
    """

    try:
        response = ollama.chat(
            model=GENERAL_MODEL_NAME,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response["message"]["content"]
    except Exception as exc:
        return (
            "General assistant is unavailable right now. "
            f"Please ensure Ollama is running and model '{GENERAL_MODEL_NAME}' is available. Error: {exc}"
        )