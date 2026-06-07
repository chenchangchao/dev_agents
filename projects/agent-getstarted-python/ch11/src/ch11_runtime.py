"""第11章服务化、缓存、容错与限流示例的公共运行时。

统一能力：
- 本章局部 data 路径
- 本地 Ollama 与云端 DeepSeek/OpenAI 兼容 API 调用
- OpenAI-compatible 响应结构
- fallback 机制，保证示例离线也能跑通

示例运行：
cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch11/src/11_1_openai_compatible_fastapi.py
"""

from __future__ import annotations

import os
import sys
import time
import uuid
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH11_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CH11_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import chat_text

BACKEND = os.getenv("LOCAL_LLM_BACKEND", "openai").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
FALLBACK_ENABLED = os.getenv("CH11_LLM_FALLBACK", "1").strip() != "0"


def data_path(*parts: str) -> Path:
    path = DATA_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def backend_name() -> str:
    if BACKEND == "ollama":
        return f"ollama:{OLLAMA_MODEL}"
    return "cloud:utils.chat_text"


def messages_to_prompt(messages: list[dict[str, str]]) -> str:
    lines = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def ask_llm(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
    label: str = "assistant",
) -> str:
    try:
        if BACKEND == "ollama":
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

        return chat_text(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
    except Exception as exc:
        if not FALLBACK_ENABLED:
            raise
        short_prompt = " ".join(prompt.split())[:180]
        return (
            f"[fallback:{label}] 当前模型后端不可用，已使用本地教学回退结果。"
            f"基于输入“{short_prompt}”，建议按服务化接口要求返回稳定、可观测、可降级的回答。"
            f" 原始异常：{type(exc).__name__}"
        )


def chat_from_messages(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 512,
    label: str = "chat",
) -> str:
    system = next((item["content"] for item in messages if item.get("role") == "system"), None)
    prompt = messages_to_prompt([item for item in messages if item.get("role") != "system"])
    return ask_llm(prompt, system=system, temperature=temperature, max_tokens=max_tokens, label=label)


def openai_chat_response(
    content: str,
    *,
    model: str | None = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
) -> dict:
    used_model = model or backend_name()
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": used_model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }
