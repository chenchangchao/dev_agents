import os
import json
import re
import sys
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import chat_text

BACKEND = os.getenv("LOCAL_LLM_BACKEND", "openai").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def backend_name() -> str:
    if BACKEND == "ollama":
        return f"ollama:{OLLAMA_MODEL}"
    return "cloud:utils.chat_text"


def ask_text(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    if BACKEND == "ollama":
        return ask_ollama_text(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    return chat_text(
        prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def ask_ollama_text(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
    response.raise_for_status()
    return response.json().get("message", {}).get("content", "").strip()


def extract_json(text: str) -> dict[str, Any]:
    """Parse a JSON object, allowing accidental wrapper text around it."""
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            preview = text[:500]
            raise ValueError(f"模型返回的JSON不完整或不合法，可能需要调大max_tokens。输出片段：{preview}") from exc


def ask_json(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> dict[str, Any]:
    if BACKEND == "ollama":
        return ask_ollama_json(
            prompt,
            system=system,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    content = ask_text(
        prompt,
        system=(system or "") + "你必须只输出合法JSON对象。",
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return extract_json(content)


def ask_ollama_json(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
    max_tokens: int = 600,
) -> dict[str, Any]:
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": False,
        "think": False,
        "format": "json",
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
    response.raise_for_status()
    content = response.json().get("message", {}).get("content", "")
    return extract_json(content)
