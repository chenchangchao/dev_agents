# 【例11-2】带基础限流的 OpenAI-compatible API
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch11/src/11_2_secure_rate_limited_api.py

import time
from collections import defaultdict, deque

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from ch11_runtime import backend_name, chat_from_messages, openai_chat_response

app = FastAPI(title="Chapter 11 Rate Limited API")
REQUEST_LOG: dict[str, deque[float]] = defaultdict(deque)
MAX_REQUESTS_PER_MINUTE = 10


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7


def check_rate_limit(client_id: str) -> bool:
    now = time.time()
    window = REQUEST_LOG[client_id]
    while window and now - window[0] > 60:
        window.popleft()
    if len(window) >= MAX_REQUESTS_PER_MINUTE:
        return False
    window.append(now)
    return True


@app.post("/v1/chat/completions")
async def chat_completions(request: Request, body: ChatRequest):
    client_id = request.client.host if request.client else "local"
    if not check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="请求过多，请稍后重试。")

    content = chat_from_messages(
        body.messages,
        temperature=body.temperature,
        max_tokens=body.max_tokens,
        label="rate_limited_api",
    )
    prompt_tokens = sum(len(item.get("content", "")) for item in body.messages)
    return openai_chat_response(content, model=body.model or backend_name(), prompt_tokens=prompt_tokens, completion_tokens=len(content))


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("服务地址：http://127.0.0.1:8012/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=8012)


if __name__ == "__main__":
    main()
