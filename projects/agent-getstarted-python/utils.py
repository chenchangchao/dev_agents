import os
from pathlib import Path
from typing import Iterable, Mapping

import dotenv
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent
dotenv.load_dotenv(PROJECT_ROOT / ".env")

DEFAULT_MODEL_ID = "gpt-4o-mini"


def get_model_id(default: str = DEFAULT_MODEL_ID) -> str:
    return os.getenv("DEEPSEEK_MODEL_ID") or os.getenv("OPENAI_MODEL_ID", default)


def get_openai_client() -> OpenAI:
    return OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("DEEPSEEK_BASE_URL") or None,
    )


def chat_completion(
    messages: Iterable[Mapping[str, str]],
    *,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs,
):
    request = {
        "model": model or get_model_id(),
        "messages": list(messages),
        "temperature": temperature,
        **kwargs,
    }
    if max_tokens is not None:
        request["max_tokens"] = max_tokens
    return get_openai_client().chat.completions.create(**request)


def chat_text(
    prompt: str,
    *,
    system: str | None = None,
    model: str | None = None,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    **kwargs,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    response = chat_completion(
        messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        **kwargs,
    )
    return response.choices[0].message.content.strip()
