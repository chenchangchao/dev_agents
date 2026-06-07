"""第5章RAG示例的公共工具。

运行示例时不直接执行本文件，而是在各脚本中导入：

cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch05/src/5_5_finance_news_rag.py

不设置 LOCAL_LLM_BACKEND 时，会走项目根目录 utils.py 中的云端DeepSeek/OpenAI兼容API。
"""

import hashlib
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import chat_text

BACKEND = os.getenv("LOCAL_LLM_BACKEND", "openai").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


@dataclass
class Document:
    page_content: str
    metadata: dict | None = None


def backend_name() -> str:
    if BACKEND == "ollama":
        return f"ollama:{OLLAMA_MODEL}"
    return "cloud:utils.chat_text"


def ask_llm(
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> str:
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

    return chat_text(
        prompt,
        system=system,
        temperature=temperature,
        max_tokens=max_tokens,
    )


def tokenize(text: str) -> list[str]:
    words = re.findall(r"[A-Za-z0-9]+", text.lower())
    chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
    chinese_bigrams = [text[i : i + 2] for i in range(len(text) - 1) if all("\u4e00" <= ch <= "\u9fff" for ch in text[i : i + 2])]
    return words + chinese_chars + chinese_bigrams


def hashing_embedding(text: str, dim: int = 256) -> list[float]:
    vec = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.md5(token.encode("utf-8")).hexdigest()
        idx = int(digest[:8], 16) % dim
        sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = math.sqrt(sum(value * value for value in vec))
    if norm:
        vec = [value / norm for value in vec]
    return vec


def cosine_similarity(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


class SimpleVectorStore:
    def __init__(self, documents: Iterable[str | Document], dim: int = 256):
        self.documents = [
            item if isinstance(item, Document) else Document(page_content=str(item))
            for item in documents
            if str(item).strip()
        ]
        self.dim = dim
        self.embeddings = [hashing_embedding(doc.page_content, dim=dim) for doc in self.documents]

    def similarity_search(self, query: str, k: int = 3) -> list[tuple[Document, float]]:
        query_vec = hashing_embedding(query, dim=self.dim)
        scored = [
            (doc, cosine_similarity(query_vec, vec))
            for doc, vec in zip(self.documents, self.embeddings)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:k]


def split_text_by_sentences(text: str, max_len: int = 120, overlap: int = 20) -> list[str]:
    sentences = [item for item in re.split(r"(?<=[。！？.!?])", text.strip()) if item.strip()]
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) <= max_len:
            current += sentence
        else:
            if current:
                chunks.append(current)
            prefix = current[-overlap:] if overlap and current else ""
            current = prefix + sentence
    if current:
        chunks.append(current)
    return chunks


def clean_text(text: str) -> str:
    text = re.sub(r"[^\u4e00-\u9fffA-Za-z0-9。！？；，、（）《》：,.!?;:()\\s-]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def print_search_results(results: list[tuple[Document, float]]):
    for index, (doc, score) in enumerate(results, start=1):
        print(f"[Top {index} | score={score:.3f}] {doc.page_content}")
