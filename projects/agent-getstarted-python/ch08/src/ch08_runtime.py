"""第8章多Agent协作示例的公共运行时。

统一能力：
- Tool基类
- 简单Message结构
- 本章局部data路径
- 本地Ollama与云端DeepSeek/OpenAI兼容API调用
- fallback机制，保证示例离线也能跑通

示例运行：
cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch08/src/8_1_role_based_multi_agent.py
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH08_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CH08_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import chat_text

BACKEND = os.getenv("LOCAL_LLM_BACKEND", "openai").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
FALLBACK_ENABLED = os.getenv("CH08_LLM_FALLBACK", "1").strip() != "0"


def data_path(*parts: str) -> Path:
    path = DATA_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def backend_name() -> str:
    if BACKEND == "ollama":
        return f"ollama:{OLLAMA_MODEL}"
    return "cloud:utils.chat_text"


class Tool:
    name = ""
    description = ""

    def __init__(self, name: str | None = None, description: str | None = None):
        if name:
            self.name = name
        if description:
            self.description = description

    def call(self, query: Any) -> str:
        raise NotImplementedError


@dataclass
class Message:
    role: str
    content: str


def format_messages(messages: list[Message]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


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
            f"基于输入“{short_prompt}”，建议综合各子Agent结果、共享状态和任务日志，"
            "输出结构化、可追溯、避免编造的总结。"
            f" 原始异常：{type(exc).__name__}"
        )
