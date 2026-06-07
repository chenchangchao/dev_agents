# 【例11-3】异步任务队列 API
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch11/src/11_3_async_task_queue_api.py
#
# 测试：
# curl -X POST http://127.0.0.1:8013/generate/ -H "Content-Type: application/json" -d '{"prompt":"你好，介绍一下你自己。"}'

import asyncio
import uuid

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from ch11_runtime import ask_llm, backend_name

app = FastAPI(title="Chapter 11 Async Task Queue")
TASKS: dict[str, dict] = {}
QUEUE: asyncio.Queue[tuple[str, str]] = asyncio.Queue()


class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 300


@app.post("/generate/")
async def generate_text(request: GenerateRequest):
    task_id = uuid.uuid4().hex
    TASKS[task_id] = {"status": "processing", "result": None, "max_tokens": request.max_tokens}
    await QUEUE.put((task_id, request.prompt))
    return {"task_id": task_id}


@app.get("/result/{task_id}")
async def get_result(task_id: str):
    return TASKS.get(task_id, {"status": "not_found", "result": None})


async def worker():
    while True:
        task_id, prompt = await QUEUE.get()
        max_tokens = TASKS[task_id]["max_tokens"]
        result = await asyncio.to_thread(
            ask_llm,
            prompt,
            system="你是异步任务队列中的文本生成助手。",
            max_tokens=max_tokens,
            label="async_task",
        )
        TASKS[task_id] = {"status": "completed", "result": result, "max_tokens": max_tokens}
        QUEUE.task_done()


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(worker())


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("服务地址：http://127.0.0.1:8013")
    uvicorn.run(app, host="0.0.0.0", port=8013)


if __name__ == "__main__":
    main()
