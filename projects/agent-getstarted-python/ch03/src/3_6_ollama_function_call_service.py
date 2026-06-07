# 【例3-6】
# ollama_function_call_service.py
#
# 本地模型示例：
# ollama list
# gemma4:e2b-mlx
# qwen3:1.7b
#
# 推荐启动：
# OLLAMA_MODEL=gemma4:e2b-mlx python3 ch03/src/3_6.py
#
# 测试：
# curl -X POST http://127.0.0.1:8081/v1/chat/function_call \
#   -H "Content-Type: application/json" \
#   -d '{"messages":[{"role":"user","content":"请告诉我北京的天气"}]}'

import os
from typing import Any

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "gemma4:e2b-mlx")
SERVER_PORT = int(os.getenv("PORT", "8081"))


app = FastAPI(title="Ollama Function Calling Demo")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    temperature: float = 0.0
    max_tokens: int = 256
    stream: bool = False


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，例如北京、上海、深圳",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculate_sum",
            "description": "计算两个数字的和",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "第一个数字"},
                    "b": {"type": "number", "description": "第二个数字"},
                },
                "required": ["a", "b"],
            },
        },
    },
]


def get_weather(city: str) -> str:
    return f"{city}的当前天气是晴，气温25°C。"


def calculate_sum(a: int | float | str, b: int | float | str) -> str:
    left = float(a)
    right = float(b)
    result = left + right
    if result.is_integer():
        result = int(result)
    return f"{a} + {b} = {result}"


def dispatch_function(name: str, arguments: dict[str, Any]) -> str:
    if name == "get_weather":
        return get_weather(str(arguments["city"]))
    if name == "calculate_sum":
        return calculate_sum(arguments["a"], arguments["b"])
    raise ValueError(f"未知函数：{name}")


def to_ollama_messages(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def call_ollama(req: ChatRequest) -> dict:
    model_name = req.model or DEFAULT_MODEL
    payload = {
        "model": model_name,
        "messages": to_ollama_messages(req.messages),
        "tools": TOOLS,
        "stream": False,
        "options": {
            "temperature": req.temperature,
            "num_predict": req.max_tokens,
        },
    }

    # Qwen3在Ollama中支持thinking；function call场景默认关闭，避免空content或额外思考文本干扰。
    if model_name.startswith("qwen3"):
        payload["think"] = False

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

    return response.json()


def build_tool_summary(tool_call: dict, tool_result: str) -> str:
    function_info = tool_call.get("function", {})
    name = function_info.get("name", "")
    arguments = function_info.get("arguments", {})
    return f"模型决定调用函数：{name}，参数：{arguments}。函数执行结果：{tool_result}"


@app.post("/v1/chat/function_call")
async def function_call(req: ChatRequest):
    if req.stream:
        raise HTTPException(status_code=400, detail="本示例暂不支持stream=true")

    data = call_ollama(req)
    message = data.get("message", {})
    tool_calls = message.get("tool_calls") or []

    if not tool_calls:
        return {
            "model": data.get("model", req.model or DEFAULT_MODEL),
            "response": message.get("content", ""),
            "raw_message": message,
        }

    results = []
    for tool_call in tool_calls:
        function_info = tool_call.get("function", {})
        name = function_info.get("name")
        arguments = function_info.get("arguments") or {}
        tool_result = dispatch_function(name, arguments)
        results.append(
            {
                "function_call": {
                    "name": name,
                    "arguments": arguments,
                },
                "tool_result": tool_result,
                "summary": build_tool_summary(tool_call, tool_result),
            }
        )

    return {
        "model": data.get("model", req.model or DEFAULT_MODEL),
        "tool_calls": results,
    }


if __name__ == "__main__":
    print(f">>> Ollama function calling model: {DEFAULT_MODEL}")
    print(f">>> API: http://127.0.0.1:{SERVER_PORT}/v1/chat/function_call")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
