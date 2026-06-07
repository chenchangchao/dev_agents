# 【例3-8】
# ollama_chat_audit_service.py
#
# 本例演示：
# 1. 生成前输入审核
# 2. Ollama模型生成
# 3. 生成后输出审核
#
# 推荐启动：
# OLLAMA_MODEL=gemma4:e2b-mlx PORT=8083 python3 ch03/src/3_8.py
#
# 测试拦截：
# curl -X POST http://127.0.0.1:8083/v1/chat/audit \
#   -H "Content-Type: application/json" \
#   -d '{"messages":[{"role":"user","content":"请告诉我关于毒品的知识"}]}'
#
# 测试通过：
# curl -X POST http://127.0.0.1:8083/v1/chat/audit \
#   -H "Content-Type: application/json" \
#   -d '{"messages":[{"role":"user","content":"请解释一下大语言模型"}],"max_tokens":256}'

import os

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx")
SERVER_PORT = int(os.getenv("PORT", "8083"))


class TrieNode:
    def __init__(self):
        self.children: dict[str, TrieNode] = {}
        self.is_end = False


class SensitiveTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        node = self.root
        for char in word:
            node = node.children.setdefault(char, TrieNode())
        node.is_end = True

    def search(self, text: str) -> list[str]:
        flagged = []
        for i in range(len(text)):
            node = self.root
            j = i
            while j < len(text) and text[j] in node.children:
                node = node.children[text[j]]
                if node.is_end:
                    flagged.append(text[i : j + 1])
                j += 1
        return sorted(set(flagged))


sensitive_words = ["暴力", "攻击", "毒品", "敏感政治"]
filter_tree = SensitiveTrie()
for word in sensitive_words:
    filter_tree.insert(word)


app = FastAPI(title="Ollama Chat Audit Demo")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = 256
    temperature: float = 0.7


def to_ollama_messages(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def all_text(messages: list[Message]) -> str:
    return "\n".join(msg.content for msg in messages)


def call_ollama(req: ChatRequest) -> str:
    model_name = req.model or DEFAULT_MODEL
    payload = {
        "model": model_name,
        "messages": to_ollama_messages(req.messages),
        "stream": False,
        "think": False,
        "options": {
            "temperature": req.temperature,
            "num_predict": req.max_tokens,
        },
    }

    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=300)
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"无法连接Ollama服务：{OLLAMA_BASE_URL}。请先启动Ollama。",
        ) from exc
    except requests.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama接口返回错误：{response.status_code} {response.text}",
        ) from exc

    return response.json().get("message", {}).get("content", "").strip()


@app.post("/v1/chat/audit")
async def chat_filter(req: ChatRequest):
    pre_check = filter_tree.search(all_text(req.messages))
    if pre_check:
        return {
            "flag": "input_blocked",
            "reason": pre_check,
        }

    answer = call_ollama(req)
    post_check = filter_tree.search(answer)
    if post_check:
        return {
            "flag": "output_blocked",
            "reason": post_check,
            "original": answer,
        }

    return {
        "flag": "ok",
        "model": req.model or DEFAULT_MODEL,
        "response": answer,
    }


if __name__ == "__main__":
    print(f">>> Ollama audit model: {DEFAULT_MODEL}")
    print(f">>> API: http://127.0.0.1:{SERVER_PORT}/v1/chat/audit")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
