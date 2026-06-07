"""第12章综合示例公共运行时。

统一能力：
- 本章局部 data 路径
- 本地 Ollama 与云端 DeepSeek/OpenAI 兼容 API 调用
- 简单 Message / ConversationContext
- fallback 机制，保证示例离线也能跑通

示例运行：
cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch12/src/12_1_model_adapter_context.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH12_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CH12_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import chat_text

BACKEND = os.getenv("LOCAL_LLM_BACKEND", "openai").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
FALLBACK_ENABLED = os.getenv("CH12_LLM_FALLBACK", "1").strip() != "0"


def data_path(*parts: str) -> Path:
    path = DATA_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def backend_name() -> str:
    if BACKEND == "ollama":
        return f"ollama:{OLLAMA_MODEL}"
    return "cloud:utils.chat_text"


@dataclass
class Message:
    role: str
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class ConversationContext:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: list[Message] = []

    def add_message(self, role: str, content: str) -> None:
        self.messages.append(Message(role, content))

    def as_dict(self) -> list[dict[str, str]]:
        return [message.to_dict() for message in self.messages]

    def text(self) -> str:
        return "\n".join(f"{message.role}: {message.content}" for message in self.messages)


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
                "options": {"temperature": temperature, "num_predict": max_tokens},
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
            f"基于输入“{short_prompt}”，建议使用本章的统一接口、工具链、MCP/A2A上下文和评估指标，"
            "给出稳定、可追溯的回答。"
            f" 原始异常：{type(exc).__name__}"
        )


def ask_with_context(
    messages: list[dict[str, str]],
    *,
    temperature: float = 0.3,
    max_tokens: int = 512,
    label: str = "context",
) -> str:
    system = next((item["content"] for item in messages if item.get("role") == "system"), None)
    prompt = "\n".join(f"{item.get('role')}: {item.get('content')}" for item in messages if item.get("role") != "system")
    return ask_llm(prompt, system=system, temperature=temperature, max_tokens=max_tokens, label=label)


def extract_city(text: str) -> str:
    for city in ["北京", "上海", "广州", "深圳"]:
        if city in text:
            return city
    return "未知"


def extract_numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?", text)


def safe_multiply(a: str, b: str) -> str:
    try:
        return str(float(a) * float(b))
    except Exception:
        return "输入格式不正确"


def ensure_text_file(path: Path, default_chunks: list[str]) -> None:
    if not path.exists():
        path.write_text("\n\n".join(default_chunks), encoding="utf-8")


def json_dumps(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False)
