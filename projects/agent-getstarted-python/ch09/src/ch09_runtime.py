"""第9章 A2A 通信示例的公共运行时。

统一能力：
- 本章局部 data 路径
- Message / PromptEntry 轻量结构
- 本地 Ollama 与云端 DeepSeek/OpenAI 兼容 API 调用
- fallback 机制，保证示例离线也能跑通

示例运行：
cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch09/src/9_1_a2a_message_dispatch.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH09_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CH09_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import chat_text

BACKEND = os.getenv("LOCAL_LLM_BACKEND", "openai").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
FALLBACK_ENABLED = os.getenv("CH09_LLM_FALLBACK", "1").strip() != "0"


def data_path(*parts: str) -> Path:
    path = DATA_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def backend_name() -> str:
    if BACKEND == "ollama":
        return f"ollama:{OLLAMA_MODEL}"
    return "cloud:utils.chat_text"


@dataclass
class PromptEntry:
    role: str
    content: str


def format_prompt(entries: list[PromptEntry]) -> str:
    return "\n".join(f"[{entry.role}] {entry.content}" for entry in entries)


def ask_llm(
    prompt: str | list[PromptEntry],
    *,
    system: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
    label: str = "assistant",
) -> str:
    prompt_text = format_prompt(prompt) if isinstance(prompt, list) else prompt
    try:
        if BACKEND == "ollama":
            messages = []
            if system:
                messages.append({"role": "system", "content": system})
            messages.append({"role": "user", "content": prompt_text})
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

        return chat_text(prompt_text, system=system, temperature=temperature, max_tokens=max_tokens)
    except Exception as exc:
        if not FALLBACK_ENABLED:
            raise
        short_prompt = " ".join(prompt_text.split())[:180]
        return (
            f"[fallback:{label}] 当前模型后端不可用，已使用本地教学回退结果。"
            f"基于输入“{short_prompt}”，建议遵循 A2A 消息协议，明确发送方、接收方、意图、"
            "上下文与权限边界，并返回可追溯的结构化结果。"
            f" 原始异常：{type(exc).__name__}"
        )
