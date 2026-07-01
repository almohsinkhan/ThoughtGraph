from __future__ import annotations

import os
from functools import lru_cache
from typing import TYPE_CHECKING

from dotenv import load_dotenv

MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_RETRIES = 2

if TYPE_CHECKING:
    from langchain_groq import ChatGroq

load_dotenv()


@lru_cache(maxsize=1)
def get_chat_model() -> ChatGroq:
    try:
        from langchain_groq import ChatGroq
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "langchain_groq is not installed. Install the backend dependencies to run the chat loop."
        ) from exc

    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError("GROQ_API_KEY is not set")

    return ChatGroq(
        model=MODEL_NAME,
        temperature=DEFAULT_TEMPERATURE,
        max_retries=DEFAULT_MAX_RETRIES,
        groq_api_key=api_key,
    )