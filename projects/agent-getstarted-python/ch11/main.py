# 第11章 智能体系统的部署、扩展与维护实战
# 【例11-1】
import os
import threading
import logging
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 Flask 应用
app = Flask(__name__)

# 加载模型和分词器
MODEL_NAME = "Qwen/Qwen3-7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).cuda()
model.eval()

# 定义锁以控制并发
model_lock = threading.Lock()

# 定义 API 路由
@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    data = request.get_json()
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    # 构建输入文本
    input_text = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            input_text += f"用户：{content}\n"
        elif role == "assistant":
            input_text += f"助手：{content}\n"

    input_text += "助手："

    # 编码输入
    inputs = tokenizer.encode(input_text, return_tensors="pt").cuda()

    # 模型生成
    with model_lock:
        outputs = model.generate(
            inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # 解码输出
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = response_text[len(input_text):].strip()

    # 构建响应
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": int(torch.time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": inputs.shape[1],
            "completion_tokens": outputs.shape[1] - inputs.shape[1],
            "total_tokens": outputs.shape[1]
        }
    }

    return jsonify(response)

# 启动 Flask 应用
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)




# 请求：
curl -X POST http://localhost:8000/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [{"role": "user", "content": "你好，介绍一下你自己。"}],
           "max_tokens": 100,
           "temperature": 0.7
         }'
# 响应：
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1714800000,
  "model": "Qwen/Qwen3-7B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好，我是由 Qwen3.0 模型驱动的智能助手，能够帮助你解答问题、提供信息和协助完成各种任务。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 30,
    "total_tokens": 50
  }
}




# 【例11-2】
import os
import ssl
import logging
from flask import Flask, request, jsonify
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.middleware.proxy_fix import ProxyFix
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化Flask应用
app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app)

# 配置请求速率限制
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per minute"]
)

# 加载模型和分词器
MODEL_NAME = "Qwen/Qwen3-7B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True).cuda()
model.eval()

# 定义API路由
@app.route('/v1/chat/completions', methods=['POST'])
@limiter.limit("10 per minute")
def chat_completions():
    data = request.get_json()
    messages = data.get("messages", [])
    max_tokens = data.get("max_tokens", 512)
    temperature = data.get("temperature", 0.7)

    if not messages:
        return jsonify({"error": "No messages provided"}), 400

    # 构建输入文本
    input_text = ""
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "user":
            input_text += f"用户：{content}\n"
        elif role == "assistant":
            input_text += f"助手：{content}\n"

    input_text += "助手："

    # 编码输入
    inputs = tokenizer.encode(input_text, return_tensors="pt").cuda()

    # 模型生成
    outputs = model.generate(
        inputs,
        max_new_tokens=max_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
    )

    # 解码输出
    response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    response_text = response_text[len(input_text):].strip()

    # 构建响应
    response = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": int(torch.time.time()),
        "model": MODEL_NAME,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": inputs.shape[1],
            "completion_tokens": outputs.shape[1] - inputs.shape[1],
            "total_tokens": outputs.shape[1]
        }
    }

    return jsonify(response)

# 启动Flask应用，启用SSL
if __name__ == '__main__':
    context = ssl.SSLContext(ssl.PROTOCOL_TLS)
    context.load_cert_chain('cert.pem', 'key.pem')
    app.run(host='0.0.0.0', port=443, ssl_context=context)




# 请求：
curl -X POST https://localhost/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "messages": [{"role": "user", "content": "你好，介绍一下你自己。"}],
           "max_tokens": 100,
           "temperature": 0.7
         }' --insecure
# 响应：
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1714800000,
  "model": "Qwen/Qwen3-7B",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "你好，我是由Qwen3.0模型驱动的智能助手，能够帮助你解答问题、提供信息和协助完成各种任务。"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 20,
    "completion_tokens": 30,
    "total_tokens": 50
  }
}




# 【例11-3】
# main.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from celery import Celery
import uvicorn
import os

app = FastAPI()

# 配置Celery
celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

class Message(BaseModel):
    prompt: str

@app.post("/generate/")
async def generate_text(message: Message):
    task = celery_app.send_task('tasks.generate_response', args=[message.prompt])
    return {"task_id": task.id}

@app.get("/result/{task_id}")
async def get_result(task_id: str):
    result = celery_app.AsyncResult(task_id)
    if result.ready():
        return {"status": "completed", "result": result.result}
    else:
        return {"status": "processing"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000)




# 任务文件：
# tasks.py
from celery import Celery
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

celery_app = Celery(
    'tasks',
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# 加载模型
model_name = "Qwen/Qwen3-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

@celery_app.task(name="tasks.generate_response")
def generate_response(prompt: str):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response



# 请求：
curl -X POST "http://localhost:8000/generate/" -H "Content-Type: application/json" -d '{"prompt":"你好，介绍一下你自己。"}'
# 响应：
{"task_id":"e3b0c44298fc1c149afbf4c8996fb924"}
# 随后，可以通过以下请求获取结果：
curl -X GET "http://localhost:8000/result/e3b0c44298fc1c149afbf4c8996fb924"
# 响应：
{"status":"completed","result":"你好，我是由Qwen3.0模型驱动的智能助手，能够帮助你解答问题、提供信息和协助完成各种任务。"}




# 【例11-4】
# -*- coding:utf-8 -*-
# 智能体多层级缓存与记忆系统构建

import os
import redis
import uuid
import chromadb
from langchain_community.chat_models import Qwen2Chat
from langchain_community.chat_models import DeepSeekChat
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import AIMessage, HumanMessage
from langchain.prompts import PromptTemplate
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 一、基础配置
REDIS_HOST = "localhost"
REDIS_PORT = 6379
CHROMA_DIR = "./chroma_db/"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 二、初始化Redis缓存连接
rds = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# 三、加载长期知识库至Chroma
loader = TextLoader("sample_knowledge.txt", encoding="utf-8")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)
split_docs = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
vectorstore = Chroma.from_documents(split_docs, embedding_model, persist_directory=CHROMA_DIR)
vectorstore.persist()

# 四、定义模型切换函数
def get_model(agent_type="qwen"):
    if agent_type == "qwen":
        return Qwen2Chat(model="Qwen/Qwen1.5-14B-Chat", device="cuda")
    else:
        return DeepSeekChat(model="deepseek-ai/deepseek-llm-chat", device="cuda")

# 五、定义对话历史缓存函数
def cache_conversation(session_id, message, role="user"):
    key = f"session:{session_id}:history"
    rds.rpush(key, f"{role}:{message}")

def get_conversation_history(session_id):
    key = f"session:{session_id}:history"
    raw = rds.lrange(key, 0, -1)
    history = []
    for entry in raw:
        decoded = entry.decode("utf-8")
        role, msg = decoded.split(":", 1)
        history.append(HumanMessage(content=msg) if role == "user" else AIMessage(content=msg))
    return history

# 六、构建会话Agent
class MemoryAgent:
    def __init__(self, session_id, agent_type="qwen"):
        self.session_id = session_id
        self.agent_type = agent_type
        self.llm = get_model(agent_type)
        self.memory = ConversationBufferMemory(return_messages=True)
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 2})
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            return_source_documents=True
        )

    def chat(self, user_input):
        history = get_conversation_history(self.session_id)
        self.memory.chat_memory.messages = history
        result = self.chain.run(user_input)
        cache_conversation(self.session_id, user_input, role="user")
        cache_conversation(self.session_id, result, role="ai")
        return result

# 七、运行示例
session_id = str(uuid.uuid4())
agent = MemoryAgent(session_id=session_id, agent_type="qwen")

print("Qwen Agent 回答:")
print(agent.chat("请问Transformer中的Self-Attention机制是怎么工作的？"))

# 切换模型后继续对话
agent2 = MemoryAgent(session_id=session_id, agent_type="deepseek")
print("\nDeepSeek Agent 回答:")
print(agent2.chat("它和多头注意力机制之间的关系是什么？"))





# 统一协议层封装版，Coze+MCP：
# -*- coding:utf-8 -*-
# MCP协议层封装的统一Agent系统，整合Coze+Redis+Chroma+Qwen3.0+Deepseek-v1

import os, uuid, redis, json
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import HumanMessage, AIMessage

# 一、统一配置
REDIS_HOST = "localhost"
REDIS_PORT = 6379
CHROMA_DIR = "./chroma_db/"
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 二、缓存系统
rds = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0)

# 三、加载向量知识库
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding)

# 四、MCP协议层封装结构
def format_mcp_messages(session_id):
    key = f"mcp:{session_id}:messages"
    raw = rds.lrange(key, 0, -1)
    return [{"role": json.loads(item.decode())["role"], "content": json.loads(item.decode())["content"]} for item in raw]

def cache_mcp_message(session_id, role, content):
    msg = json.dumps({"role": role, "content": content})
    key = f"mcp:{session_id}:messages"
    rds.rpush(key, msg)

# 五、模型统一封装为Coze调用函数
def call_model(session_id, user_input, agent_type="qwen"):
    memory = ConversationBufferMemory(return_messages=True)
    history = []

    # 构建统一上下文格式（模拟Coze内消息格式）
    mcp_history = format_mcp_messages(session_id)
    for entry in mcp_history:
        if entry["role"] == "user":
            history.append(HumanMessage(content=entry["content"]))
        else:
            history.append(AIMessage(content=entry["content"]))
    memory.chat_memory.messages = history

    llm = Qwen2Chat(model="Qwen/Qwen1.5-14B-Chat") if agent_type == "qwen" else DeepSeekChat(model="deepseek-ai/deepseek-llm-chat")

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 2}),
        memory=memory
    )

    response = qa_chain.run(user_input)
    cache_mcp_message(session_id, "user", user_input)
    cache_mcp_message(session_id, "assistant", response)
    return response

# 六、模拟Coze平台任务流：Agent切换对话
def simulate_coze_agent_workflow():
    session_id = str(uuid.uuid4())
    print("[Qwen Agent 回复]")
    print(call_model(session_id, "什么是位置编码？", agent_type="qwen"))

    print("\n[Deepseek Agent 回复]")
    print(call_model(session_id, "位置编码和RoPE有什么不同？", agent_type="deepseek"))

# 七、运行模拟工作流
simulate_coze_agent_workflow()




# 【例11-5】
# -*- coding:utf-8 -*-
# Qwen3.0 + Deepseek-v1 智能体容错机制实现框架

import os
import time
import uuid
import logging
import traceback
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import ConversationChain

# 一、日志配置
logging.basicConfig(
    filename="agent_error.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# 二、模型封装
def load_model(agent_type="qwen"):
    if agent_type == "qwen":
        return Qwen2Chat(model="Qwen/Qwen1.5-7B-Chat", device="cuda")
    else:
        return DeepSeekChat(model="deepseek-ai/deepseek-llm-chat", device="cuda")

# 三、封装调用类
class ResilientAgent:
    def __init__(self, primary="qwen", backup="deepseek", max_retries=2, timeout=15):
        self.primary_type = primary
        self.backup_type = backup
        self.max_retries = max_retries
        self.timeout = timeout
        self.session_id = str(uuid.uuid4())
        self.memory = ConversationBufferMemory(return_messages=True)

    def _call_with_timeout(self, model, input_text):
        from concurrent.futures import ThreadPoolExecutor, TimeoutError
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._safe_run_chain, model, input_text)
            return future.result(timeout=self.timeout)

    def _safe_run_chain(self, model, input_text):
        chain = ConversationChain(llm=model, memory=self.memory)
        return chain.run(input_text)

    def _log_failure(self, error_msg):
        logging.error(f"Session {self.session_id} Failed:\n{error_msg}")

    def ask(self, user_input):
        self.memory.chat_memory.add_message(HumanMessage(content=user_input))
        models_to_try = [self.primary_type, self.backup_type]

        for model_type in models_to_try:
            retries = 0
            while retries < self.max_retries:
                try:
                    model = load_model(model_type)
                    print(f"[INFO] 尝试使用模型：{model_type}（第{retries+1}次）")
                    response = self._call_with_timeout(model, user_input)
                    self.memory.chat_memory.add_message(AIMessage(content=response))
                    return response
                except Exception as e:
                    retries += 1
                    err = traceback.format_exc()
                    self._log_failure(err)
                    print(f"[WARN] 模型{model_type}调用失败，错误：{str(e)}")

        # 所有模型都失败，返回兜底响应
        fallback_msg = "很抱歉，我暂时无法回答您的问题，请稍后再试。"
        self.memory.chat_memory.add_message(AIMessage(content=fallback_msg))
        return fallback_msg

# 四、运行示例
def run_demo():
    agent = ResilientAgent(primary="qwen", backup="deepseek", max_retries=2, timeout=10)

    print("[用户] 请解释注意力机制的原理")
    reply1 = agent.ask("请解释注意力机制的原理")
    print("[智能体回复]", reply1)

    print("\n[用户] 它与Transformer的关系呢？")
    reply2 = agent.ask("它与Transformer的关系呢？")
    print("[智能体回复]", reply2)

    print("\n[用户] 多头注意力怎么提升模型性能？")
    reply3 = agent.ask("多头注意力怎么提升模型性能？")
    print("[智能体回复]", reply3)

run_demo()




# 【例11-6】
# -*- coding:utf-8 -*-
# Qwen3.0/Deepseek智能体限流与降级系统

import asyncio
import time
import uuid
import redis
from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain.memory import ConversationBufferMemory
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import ConversationChain

# 一、Redis配置与令牌桶限流
redis_client = redis.Redis(host="localhost", port=6379, db=0)
BUCKET_KEY = "agent_rate_limit"
MAX_TOKENS = 10
REFILL_RATE = 1  # 每秒补充1个token

def refill_bucket():
    now = int(time.time())
    last_refill = int(redis_client.get("last_refill") or now)
    elapsed = now - last_refill
    if elapsed > 0:
        tokens = min(MAX_TOKENS, int(redis_client.get(BUCKET_KEY) or 0) + elapsed * REFILL_RATE)
        redis_client.set(BUCKET_KEY, tokens)
        redis_client.set("last_refill", now)

def acquire_token():
    refill_bucket()
    tokens = int(redis_client.get(BUCKET_KEY) or 0)
    if tokens > 0:
        redis_client.decr(BUCKET_KEY)
        return True
    return False

# 二、请求体结构
class ChatRequest(BaseModel):
    session_id: str
    message: str
    priority: int = 1  # 1为默认，0为高优先级

# 三、模型加载函数
def get_model(agent_type="qwen"):
    if agent_type == "qwen":
        return Qwen2Chat(model="Qwen/Qwen1.5-7B-Chat")
    return DeepSeekChat(model="deepseek-ai/deepseek-llm-chat")

# 四、智能体服务封装（含降级策略）
async def handle_request(session_id, message):
    memory = ConversationBufferMemory(return_messages=True)
    memory.chat_memory.add_message(HumanMessage(content=message))

    # 主模型调用（Qwen）
    try:
        model = get_model("qwen")
        chain = ConversationChain(llm=model, memory=memory)
        result = await asyncio.to_thread(chain.run, message)
        memory.chat_memory.add_message(AIMessage(content=result))
        return result
    except Exception as e:
        print("[WARN] Qwen失败，尝试降级至Deepseek")
        # 备份模型调用（Deepseek）
        try:
            model = get_model("deepseek")
            chain = ConversationChain(llm=model, memory=memory)
            result = await asyncio.to_thread(chain.run, message)
            memory.chat_memory.add_message(AIMessage(content=result))
            return result
        except Exception as e2:
            print("[ERROR] Deepseek也失败，返回兜底响应")
            return "系统繁忙，请稍后再试。"

# 五、FastAPI接口服务
app = FastAPI()
request_queue = asyncio.PriorityQueue()

@app.post("/chat")
async def chat(req: ChatRequest):
    if not acquire_token():
        raise HTTPException(status_code=429, detail="请求过多，请稍后重试。")

    fut = asyncio.get_event_loop().create_future()
    await request_queue.put((req.priority, time.time(), req, fut))
    return await fut

# 六、并发消费者处理队列请求
async def queue_worker():
    while True:
        if not request_queue.empty():
            _, _, req, fut = await request_queue.get()
            try:
                result = await handle_request(req.session_id, req.message)
                fut.set_result({"session_id": req.session_id, "response": result})
            except Exception as ex:
                fut.set_result({"session_id": req.session_id, "response": "系统异常"})
        await asyncio.sleep(0.1)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(queue_worker())

# 七、运行方法（通过`uvicorn script:app --reload`启动）
