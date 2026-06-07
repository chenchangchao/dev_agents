# 【例3-9】
# ollama_chat_with_confidence_proxy.py
#
# 原Transformers版本可以通过output_scores拿到每步token概率。
# Ollama HTTP API默认不返回logits/top-k概率，因此本例改为返回可观测的生成质量信息：
# - done_reason
# - prompt_eval_count
# - eval_count
# - total_duration
# - tokens_per_second
# - confidence_proxy
#
# 推荐启动：
# OLLAMA_MODEL=gemma4:e2b-mlx PORT=8084 python3 ch03/src/3_9.py
#
# 测试：
# curl -X POST http://127.0.0.1:8084/v1/chat/with_confidence \
#   -H "Content-Type: application/json" \
#   -d '{"messages":[{"role":"user","content":"请解释一下LangChain的核心组件"}],"max_tokens":256}'

import os

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx")
SERVER_PORT = int(os.getenv("PORT", "8084"))

app = FastAPI(title="Ollama Confidence Proxy Demo")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    max_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 5


def to_ollama_messages(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def build_payload(req: ChatRequest) -> dict:
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
    return payload


def call_ollama(req: ChatRequest) -> dict:
    try:
        response = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=build_payload(req), timeout=300)
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
    return response.json()


def tokens_per_second(eval_count: int, eval_duration_ns: int) -> float:
    if not eval_count or not eval_duration_ns:
        return 0.0
    return eval_count / (eval_duration_ns / 1_000_000_000)


def confidence_proxy(data: dict, response_text: str) -> dict:
    done_reason = data.get("done_reason", "")
    eval_count = int(data.get("eval_count") or 0)

    if not response_text:
        level = "low"
        reason = "模型没有返回有效内容。"
    elif done_reason == "length":
        level = "medium"
        reason = "回答因为max_tokens限制被截断，建议增大max_tokens后再评估。"
    elif done_reason == "stop" and eval_count >= 20:
        level = "high"
        reason = "模型正常停止且生成内容长度较充分。"
    else:
        level = "medium"
        reason = "模型已生成回答，但可结合人工评测继续确认质量。"

    return {
        "level": level,
        "reason": reason,
        "note": "Ollama未返回逐token logits，本字段是基于结束原因和生成长度的近似质量提示，不等同于真实概率置信度。",
    }


@app.post("/v1/chat/with_confidence")
async def chat_with_confidence(req: ChatRequest):
    data = call_ollama(req)
    message = data.get("message", {})
    response_text = message.get("content", "").strip()
    eval_count = int(data.get("eval_count") or 0)
    eval_duration = int(data.get("eval_duration") or 0)

    return {
        "model": data.get("model", req.model or DEFAULT_MODEL),
        "response": response_text,
        "confidence_proxy": confidence_proxy(data, response_text),
        "generation_stats": {
            "done_reason": data.get("done_reason"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": eval_count,
            "total_duration_ns": data.get("total_duration"),
            "load_duration_ns": data.get("load_duration"),
            "prompt_eval_duration_ns": data.get("prompt_eval_duration"),
            "eval_duration_ns": eval_duration,
            "tokens_per_second": round(tokens_per_second(eval_count, eval_duration), 2),
        },
        "token_confidence": {
            "available": False,
            "requested_top_k": req.top_k,
            "reason": "Ollama /api/chat 不返回 Transformers output_scores，因此不能计算真实top-k token概率。",
        },
    }


if __name__ == "__main__":
    print(f">>> Ollama confidence proxy model: {DEFAULT_MODEL}")
    print(f">>> API: http://127.0.0.1:{SERVER_PORT}/v1/chat/with_confidence")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
