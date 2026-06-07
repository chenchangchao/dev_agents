# 【例11-6】限流、优先级队列与降级服务
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch11/src/11_6_rate_limit_queue_service.py

import asyncio
import time

import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ch11_runtime import ask_llm, backend_name

MAX_TOKENS = 10
REFILL_RATE = 1
BUCKET = {"tokens": MAX_TOKENS, "last_refill": time.time()}

app = FastAPI(title="Chapter 11 Rate Limit Queue Service")
request_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()


class ChatRequest(BaseModel):
    session_id: str
    message: str
    priority: int = 1


def refill_bucket() -> None:
    now = time.time()
    elapsed = now - BUCKET["last_refill"]
    if elapsed > 0:
        BUCKET["tokens"] = min(MAX_TOKENS, BUCKET["tokens"] + elapsed * REFILL_RATE)
        BUCKET["last_refill"] = now


def acquire_token() -> bool:
    refill_bucket()
    if BUCKET["tokens"] >= 1:
        BUCKET["tokens"] -= 1
        return True
    return False


async def handle_request(session_id: str, message: str) -> str:
    try:
        return await asyncio.to_thread(
            ask_llm,
            message,
            system=f"你是限流队列服务中的助手，会话ID：{session_id}。",
            max_tokens=500,
            label="rate_limit_queue",
        )
    except Exception:
        return "系统繁忙，请稍后再试。"


@app.post("/chat")
async def chat(req: ChatRequest):
    if not acquire_token():
        raise HTTPException(status_code=429, detail="请求过多，请稍后重试。")

    future = asyncio.get_event_loop().create_future()
    await request_queue.put((req.priority, time.time(), req, future))
    return await future


async def queue_worker():
    while True:
        _, _, req, future = await request_queue.get()
        result = await handle_request(req.session_id, req.message)
        if not future.done():
            future.set_result({"session_id": req.session_id, "response": result})
        request_queue.task_done()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(queue_worker())


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("服务地址：http://127.0.0.1:8016/chat")
    uvicorn.run(app, host="0.0.0.0", port=8016)


if __name__ == "__main__":
    main()
