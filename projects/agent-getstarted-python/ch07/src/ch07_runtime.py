"""第7章示例的公共运行时封装。

统一能力：
- Tool基类
- 本章局部 data 路径
- 本地Ollama与云端DeepSeek/OpenAI兼容API调用
- 简单上下文段、窗口记忆和fallback机制

示例运行：
cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch07/src/7_1_stateful_agent_lifecycle.py

不设置 LOCAL_LLM_BACKEND 时，会调用项目根目录 utils.py。
模型不可用时默认启用教学fallback，保证脚本可离线跑通。
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH07_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CH07_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import chat_text

BACKEND = os.getenv("LOCAL_LLM_BACKEND", "openai").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
FALLBACK_ENABLED = os.getenv("CH07_LLM_FALLBACK", "1").strip() != "0"


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
class ContextSegment:
    role: str
    content: str
    type: str
    timestamp: float
    priority: float = 0.5


def build_segment(role: str, content: str, segment_type: str, priority: float | None = None) -> dict[str, Any]:
    return {
        "role": role,
        "content": content,
        "type": segment_type,
        "timestamp": time.time(),
        "priority": 1.0 if priority is None and segment_type == "system" else (priority if priority is not None else 0.5),
        "length": max(1, len(content) // 2),
    }


class ConversationWindowMemory:
    def __init__(self, k: int = 3):
        self.k = k
        self.messages: list[tuple[str, str]] = []

    def add(self, role: str, content: str) -> None:
        self.messages.append((role, content))
        self.messages = self.messages[-self.k * 2 :]

    def text(self) -> str:
        return "\n".join(f"{role}: {content}" for role, content in self.messages)


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
        short_prompt = " ".join(prompt.split())[:160]
        return (
            f"[fallback:{label}] 当前模型后端不可用，已使用本地教学回退结果。"
            f"基于输入“{short_prompt}”，建议结合状态、工具结果、历史上下文和错误信息，"
            "给出谨慎、结构化、可追溯的回答。"
            f" 原始异常：{type(exc).__name__}"
        )
