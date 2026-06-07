# 【例11-1】OpenAI-compatible Chat API 服务
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch11/src/11_1_openai_compatible_fastapi.py
#
# 测试：
# curl -X POST http://127.0.0.1:8011/v1/chat/completions \
#   -H "Content-Type: application/json" \
#   -d '{"messages":[{"role":"user","content":"你好，介绍一下你自己。"}],"max_tokens":120}'

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from ch11_runtime import backend_name, chat_from_messages, openai_chat_response

app = FastAPI(title="Chapter 11 OpenAI-Compatible API")


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[dict[str, str]]
    max_tokens: int = 512
    temperature: float = 0.7


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatRequest):
    content = chat_from_messages(
        request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        label="openai_api",
    )
    prompt_tokens = sum(len(item.get("content", "")) for item in request.messages)
    return openai_chat_response(
        content,
        model=request.model or backend_name(),
        prompt_tokens=prompt_tokens,
        completion_tokens=len(content),
    )


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("服务地址：http://127.0.0.1:8011/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=8011)


if __name__ == "__main__":
    main()
