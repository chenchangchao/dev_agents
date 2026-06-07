# 【例3-5】
# qwen3_colab_fastapi_service.py
#
# 本地Mac + Ollama推荐运行：
# ollama pull qwen3:1.7b
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=qwen3:1.7b python3 ch03/src/3_5.py
# 如需查看Qwen3 thinking过程，可额外设置：OLLAMA_THINK=true
#
# Colab T4推荐安装：
# !pip uninstall -y torchvision torchao
# !pip install -U torch transformers accelerate fastapi uvicorn
# 运行时如果刚卸载/升级过torch相关包，请重启Colab Runtime。
# 使用Colab/Transformers后端前设置：
# import os
# os.environ["LOCAL_LLM_BACKEND"] = "transformers"
#
# Colab Notebook内测试：
# from fastapi.testclient import TestClient
# client = TestClient(app)
# response = client.post("/v1/chat/completions", json=sample_payload())
# print(response.json()["choices"][0]["message"]["content"])

import json
import os
import threading
import time
from importlib import metadata
from typing import Generator

import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

MODEL_ID = "Qwen/Qwen3-1.7B"
BACKEND = os.getenv("LOCAL_LLM_BACKEND", "ollama").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:1.7b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
OLLAMA_THINK = os.getenv("OLLAMA_THINK", "false").strip().lower() in {"1", "true", "yes", "on"}
SERVER_PORT = int(os.getenv("PORT", "8080"))
TORCHAO_MIN_VERSION = (0, 16, 0)

tokenizer = None
model = None
torch_runtime = None


def parse_version(version: str) -> tuple[int, ...]:
    parts = []
    for part in version.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def load_transformers_dependencies():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError(
            "当前选择的是 LOCAL_LLM_BACKEND=transformers，"
            "需要先安装 torch 和 transformers。"
        ) from exc
    return torch, AutoModelForCausalLM, AutoTokenizer


def check_colab_environment(torch_module):
    if not torch_module.cuda.is_available():
        raise RuntimeError("未检测到CUDA GPU。请在Colab中启用 Runtime > Change runtime type > T4 GPU。")

    try:
        torchao_version = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        torchao_version = None

    if torchao_version and parse_version(torchao_version) < TORCHAO_MIN_VERSION:
        raise RuntimeError(
            "检测到 Colab 中的 torchao 版本过旧："
            f"{torchao_version}。本示例不需要 torchao，"
            "请先执行：!pip uninstall -y torchao，然后重启 Colab Runtime。"
        )

    print(">>> CUDA GPU:", torch_module.cuda.get_device_name(0))


def load_model_once():
    global tokenizer, model, torch_runtime
    if tokenizer is not None and model is not None:
        return tokenizer, model

    torch_runtime, auto_model_cls, auto_tokenizer_cls = load_transformers_dependencies()
    check_colab_environment(torch_runtime)
    print(">>> 正在加载模型与Tokenizer...")
    tokenizer = auto_tokenizer_cls.from_pretrained(MODEL_ID)
    model = auto_model_cls.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch_runtime.float16,
    )
    model.eval()
    print(">>> 模型加载完成")
    return tokenizer, model


# Step 1: 构造FastAPI服务与数据结构
app = FastAPI(title="Qwen3 Local OpenAI-Compatible API")


class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = False


def current_model_id() -> str:
    return OLLAMA_MODEL if BACKEND == "ollama" else MODEL_ID


def message_dicts(messages: list[Message]) -> list[dict[str, str]]:
    return [{"role": msg.role, "content": msg.content} for msg in messages]


def build_inputs(messages: list[Message]):
    tokenizer_obj, model_obj = load_model_once()
    return tokenizer_obj.apply_chat_template(
        message_dicts(messages),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model_obj.device)


def transformers_generate_text(messages: list[Message], max_tokens: int, temperature: float) -> str:
    if torch_runtime is None:
        load_model_once()
    tokenizer_obj, model_obj = load_model_once()
    inputs = build_inputs(messages)
    do_sample = temperature > 0

    with torch_runtime.no_grad():
        output_ids = model_obj.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            pad_token_id=tokenizer_obj.eos_token_id,
        )

    input_len = inputs["input_ids"].shape[-1]
    new_tokens = output_ids[0][input_len:]
    return tokenizer_obj.decode(new_tokens, skip_special_tokens=True).strip()


def ollama_generate_text(messages: list[Message], max_tokens: int, temperature: float, model_name: str | None) -> str:
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model_name or OLLAMA_MODEL,
                "messages": message_dicts(messages),
                "stream": False,
                "think": OLLAMA_THINK,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
            timeout=300,
        )
        response.raise_for_status()
    except requests.ConnectionError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"无法连接 Ollama 服务：{OLLAMA_BASE_URL}。请先启动 Ollama。原始错误：{exc}",
        ) from exc
    except requests.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama接口返回错误：{response.status_code} {response.text}",
        ) from exc

    data = response.json()
    message = data.get("message", {})
    content = message.get("content", "").strip()
    if content:
        return content
    thinking = message.get("thinking", "").strip()
    if thinking:
        return (
            "模型只生成了 thinking 内容，还没有生成最终回答。"
            "请调大 max_tokens，或保持 OLLAMA_THINK=false。"
        )
    return ""


def generate_text(messages: list[Message], max_tokens: int, temperature: float, model_name: str | None = None) -> str:
    if BACKEND == "ollama":
        return ollama_generate_text(messages, max_tokens, temperature, model_name)
    if BACKEND == "transformers":
        return transformers_generate_text(messages, max_tokens, temperature)
    raise ValueError("LOCAL_LLM_BACKEND 只支持 ollama 或 transformers")


def ollama_stream_response(
    messages: list[Message],
    max_tokens: int,
    temperature: float,
    model_name: str | None,
) -> Generator[str, None, None]:
    try:
        with requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json={
                "model": model_name or OLLAMA_MODEL,
                "messages": message_dicts(messages),
                "stream": True,
                "think": OLLAMA_THINK,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                },
            },
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


def stream_response(
    messages: list[Message],
    max_tokens: int,
    temperature: float,
    model_name: str | None = None,
) -> Generator[str, None, None]:
    if BACKEND == "ollama":
        yield from ollama_stream_response(messages, max_tokens, temperature, model_name)
        return

    response = transformers_generate_text(messages, max_tokens, temperature)
    for i in range(0, len(response), 10):
        chunk = {
            "choices": [
                {
                    "delta": {
                        "content": response[i:i + 10],
                    }
                }
            ]
        }
        yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"
        time.sleep(0.03)
    yield "data: [DONE]\n\n"


# Step 2: OpenAI兼容响应接口
@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest):
    response_model = req.model or current_model_id()
    if req.stream:
        return StreamingResponse(
            stream_response(req.messages, req.max_tokens, req.temperature, response_model),
            media_type="text/event-stream",
        )

    result = generate_text(req.messages, req.max_tokens, req.temperature, response_model)
    return {
        "id": "chatcmpl-local-qwen3",
        "object": "chat.completion",
        "model": response_model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": result,
                },
                "finish_reason": "stop",
            }
        ],
    }


def sample_payload(stream: bool = False) -> dict:
    return {
        "model": current_model_id(),
        "messages": [
            {"role": "system", "content": "你是一名专业中文技术助手。"},
            {"role": "user", "content": "请解释一下LoRA技术的基本原理"},
        ],
        "temperature": 0.7,
        "max_tokens": 200,
        "stream": stream,
    }


def notebook_test():
    from fastapi.testclient import TestClient

    client = TestClient(app)
    response = client.post("/v1/chat/completions", json=sample_payload(stream=False))
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    print(f">>> TestClient响应，backend={BACKEND}，model={current_model_id()}：")
    print(content)


def start_colab_server(host: str = "0.0.0.0", port: int = 8080):
    """Start uvicorn in a background thread for Colab/Jupyter notebooks."""
    thread = threading.Thread(
        target=uvicorn.run,
        kwargs={
            "app": app,
            "host": host,
            "port": port,
            "log_level": "info",
        },
        daemon=True,
    )
    thread.start()
    print(f">>> Uvicorn后台服务已启动：http://127.0.0.1:{port}")
    return thread


# Step 3: 启动服务
# Colab/Jupyter中不要直接调用 uvicorn.run(app, ...)，否则可能遇到
# "asyncio.run() cannot be called from a running event loop"。
# 推荐：
# notebook_test()
# 或者：
# server_thread = start_colab_server(port=8080)
if __name__ == "__main__":
    print(f">>> backend={BACKEND}，model={current_model_id()}")
    print(f">>> OpenAI兼容接口：http://127.0.0.1:{SERVER_PORT}/v1/chat/completions")
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
