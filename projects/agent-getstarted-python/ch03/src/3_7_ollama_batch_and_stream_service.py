# 【例3-7】
# ollama_batch_and_stream_service.py
#
# 本例演示：
# 1. 批量对话请求
# 2. Ollama原生流式输出
#
# 推荐启动：
# OLLAMA_MODEL=gemma4:e2b-mlx python3 ch03/src/3_7.py
#
# 测试批处理：
# curl -X POST http://127.0.0.1:8082/v1/chat/batch \
#   -H "Content-Type: application/json" \
#   -d '{"messages_list":[[{"role":"user","content":"介绍一下LangChain"}],[{"role":"user","content":"什么是LoRA技术？"}]],"max_tokens":256}'
#
# 测试流式：
# curl -N -X POST http://127.0.0.1:8082/v1/chat/stream \
#   -H "Content-Type: application/json" \
#   -d '{"messages":[{"role":"user","content":"请解释一下大语言模型的上下文窗口"}],"max_tokens":256}'

import json
import os
from typing import Generator

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx")
SERVER_PORT = int(os.getenv("PORT", "8082"))

app = FastAPI(title="Ollama Batch and Stream Demo")


class ChatItem(BaseModel):
    role: str
    content: str


class BatchRequest(BaseModel):
    model: str | None = None
    messages_list: list[list[ChatItem]]
    max_tokens: int = 256
    temperature: float = 0.7


class StreamRequest(BaseModel):
    model: str | None = None
    messages: list[ChatItem]
    max_tokens: int = 256
    temperature: float = 0.7


def to_ollama_messages(messages: list[ChatItem]) -> list[dict[str, str]]:
    return [{"role": item.role, "content": item.content} for item in messages]


def build_payload(
    model_name: str,
    messages: list[ChatItem],
    max_tokens: int,
    temperature: float,
    stream: bool,
) -> dict:
    payload = {
        "model": model_name,
        "messages": to_ollama_messages(messages),
        "stream": stream,
        "think": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
        },
    }
    return payload


def raise_ollama_error(response: requests.Response):
    try:
        response.raise_for_status()
    except requests.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama接口返回错误：{response.status_code} {response.text}",
        ) from exc


def chat_once(model_name: str, messages: list[ChatItem], max_tokens: int, temperature: float) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=build_payload(model_name, messages, max_tokens, temperature, stream=False),
            timeout=300,
        )
    except requests.ConnectionError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"无法连接Ollama服务：{OLLAMA_BASE_URL}。请先启动Ollama。",
        ) from exc

    raise_ollama_error(response)
    data = response.json()
    return data.get("message", {}).get("content", "").strip()


@app.post("/v1/chat/batch")
async def batch_chat(req: BatchRequest):
    model_name = req.model or DEFAULT_MODEL
    results = []
    for index, messages in enumerate(req.messages_list):
        answer = chat_once(model_name, messages, req.max_tokens, req.temperature)
        results.append(
            {
                "index": index,
                "response": answer,
            }
        )
    return {
        "model": model_name,
        "results": results,
    }


def stream_generator(req: StreamRequest) -> Generator[str, None, None]:
    model_name = req.model or DEFAULT_MODEL
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=build_payload(model_name, req.messages, req.max_tokens, req.temperature, stream=True),
            stream=True,
            timeout=300,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line:
                    continue
                data = json.loads(line.decode("utf-8"))
                content = data.get("message", {}).get("content", "")
                if content:
                    chunk = {"choices": [{"delta": {"content": content}}]}
                    yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
                if data.get("done"):
                    break
    except requests.RequestException as exc:
        chunk = {"error": f"Ollama流式接口调用失败：{exc}"}
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
    yield "data: [DONE]\n\n"


@app.post("/v1/chat/stream")
async def stream_chat(req: StreamRequest):
    return StreamingResponse(stream_generator(req), media_type="text/event-stream")


if __name__ == "__main__":
    print(f">>> Ollama batch/stream model: {DEFAULT_MODEL}")
    print(f">>> Batch API: http://127.0.0.1:{SERVER_PORT}/v1/chat/batch")
    print(f">>> Stream API: http://127.0.0.1:{SERVER_PORT}/v1/chat/stream")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
