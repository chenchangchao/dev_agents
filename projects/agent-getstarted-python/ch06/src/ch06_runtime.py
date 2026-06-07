"""第6章示例的公共运行时封装。

本文件统一管理：
- Tool基类
- 结构化上下文段
- ch06/data局部持久化路径
- 本地Ollama与云端DeepSeek/OpenAI兼容API调用

示例运行：
cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch06/src/6_1_structured_context_fusion.py

不设置 LOCAL_LLM_BACKEND 时，会调用项目根目录 utils.py 中的云端API封装。
如果本地/云端模型不可用，默认返回教学用fallback文本，保证脚本可离线跑通。
"""

from __future__ import annotations

import json
import os
import sys
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CH06_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CH06_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import chat_text

BACKEND = os.getenv("LOCAL_LLM_BACKEND", "openai").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
FALLBACK_ENABLED = os.getenv("CH06_LLM_FALLBACK", "1").strip() != "0"


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
    task_id: str = "default"
    model: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_segment(
    role: str,
    content: str,
    segment_type: str,
    task_id: str = "default",
    *,
    model: str = "",
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return ContextSegment(
        role=role,
        content=content,
        type=segment_type,
        task_id=task_id,
        model=model,
        meta=meta or {},
    ).to_dict()


def format_segments(segments: list[dict[str, Any]], limit: int | None = None) -> str:
    selected = segments[-limit:] if limit else segments
    return "\n".join(
        f"[{seg.get('type', '').upper()}] {seg.get('role', '')}: {seg.get('content', '')}"
        for seg in selected
    )


def ask_llm(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.2,
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
        return fallback_answer(prompt, system=system, label=label, error=exc)


def fallback_answer(prompt: str, *, system: str | None, label: str, error: Exception) -> str:
    short_prompt = " ".join(prompt.split())[:160]
    role_hint = system or "宏观经济与政策分析助手"
    return (
        f"[fallback:{label}] 当前模型后端不可用，已使用本地教学回退结果。"
        f"角色设定：{role_hint}。基于输入“{short_prompt}”，"
        "可给出简要判断：应结合政策背景、结构化数据、工具观测与历史上下文，"
        "形成谨慎、可追溯、避免编造的回答。"
        f" 原始异常：{type(error).__name__}"
    )


class JsonlContextLog:
    def __init__(self, filename: str):
        self.path = data_path(filename)

    def append(self, segment: dict[str, Any]) -> None:
        with self.path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(segment, ensure_ascii=False) + "\n")

    def tail(self, n: int = 5) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()[-n:]
        return [json.loads(line) for line in lines if line.strip()]


class PromptCache:
    def __init__(self, filename: str):
        self.path = data_path(filename)

    def load(self) -> dict[str, str]:
        if not self.path.exists():
            return {}
        return json.loads(self.path.read_text(encoding="utf-8"))

    def save(self, cache: dict[str, str]) -> None:
        self.path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
