# 第3章 大模型开发基础
# 【例3-1】
# openai_saas_example.py

import openai
import os
import json
# import requests  # 可选：如果需要直接调用HTTP接口
from typing import List, Dict
import dotenv  # 加载环境变量

dotenv.load_dotenv()  # 从.env文件加载环境变量
import os

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL_ID = os.getenv("DEEPSEEK_MODEL_ID")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

# 设置OpenAI的API Key
openai.api_key = DEEPSEEK_API_KEY
openai.base_url = DEEPSEEK_BASE_URL

# 封装消息构造函数
def build_message(user_prompt: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": "你是一位语言学家，擅长解释中文成语的典故、来历与含义"},
        {"role": "user", "content": user_prompt}
    ]

# 调用OpenAI Chat API函数
def chat_with_openai(prompt: str, temperature: float = 0.5, max_tokens: int = 300) -> str:
    messages = build_message(prompt)

    try:
        response = openai.ChatCompletion.create(
            model=DEEPSEEK_MODEL_ID,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        reply = response['choices'][0]['message']['content']
        return reply.strip()

    except Exception as e:
        return f"调用失败: {str(e)}"

# 示例调用函数：请求多个成语解释
def batch_inference():
    idioms = [
        "破釜沉舟", "卧薪尝胆", "指鹿为马", "望梅止渴", "画龙点睛"
    ]
    results = {}
    for idiom in idioms:
        reply = chat_with_openai(f"请解释成语“{idiom}”的来历和含义")
        results[idiom] = reply
    return results

# 主程序
if __name__ == "__main__":
    print(">>> 开始批量调用 OpenAI ChatCompletion 接口")
    result = batch_inference()
    for idiom, explanation in result.items():
        print(f"\n【{idiom}】\n{explanation}\n")




# 【例3-2】
# qwen3_local_deploy.py

# 第一步：安装依赖
# pip install torch torchvision transformers accelerate
# pip install -U qwen==1.0.3  # 确保qwen库为官方发布版本

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# 第二步：加载Qwen-235B模型
print(">>> 正在加载Tokenizer与模型...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-235B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen-235B",
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True
)
model.eval()
print(">>> 模型加载完毕")

# 第三步：构造输入
def build_prompt(user_input: str) -> str:
    return f"请解释以下中文成语的含义与出处：{user_input}"

# 第四步：执行推理
def run_inference(prompt: str, max_tokens: int = 128, temperature: float = 0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature
        )
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded

# 第五步：批量测试
idioms = ["画蛇添足", "纸上谈兵", "掩耳盗铃", "望梅止渴", "指鹿为马"]
print(">>> 开始批量推理任务")

for idiom in idioms:
    prompt = build_prompt(idiom)
    result = run_inference(prompt)
    print(f"\n【{idiom}】\n{result.strip()}\n")




# 【例3-3】
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from peft import get_peft_model, LoraConfig, TaskType
from datasets import Dataset

# Step 1: 加载Qwen模型与Tokenizer
model_name = "Qwen/Qwen-1.8B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)

# Step 2: 构造LoRA注入配置
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["q_proj", "v_proj"]
)

# Step 3: 注入LoRA模块
model = get_peft_model(base_model, lora_config)

# Step 4: 构造真实中文观点分类数据
data_dict = {
    "text": [
        "客服态度很好，处理问题及时",
        "屏幕亮度太低，看起来很费眼",
        "物流太快了，包装也非常结实",
        "使用几天就死机了，非常糟糕",
        "价格优惠，功能也很全面",
        "音质不清晰，续航时间短"
    ],
    "label": ["正面", "负面", "正面", "负面", "正面", "负面"]
}
dataset = Dataset.from_dict(data_dict)

# Step 5: 数据预处理与Prompt构造
def preprocess(sample):
    prompt = f"请判断以下句子的情感倾向：{sample['text']} 回答：{sample['label']}"
    return tokenizer(prompt, truncation=True, padding="max_length", max_length=128)

tokenized_dataset = dataset.map(preprocess)

# Step 6: 配置训练参数
train_args = TrainingArguments(
    output_dir="./output/qwen_lora_sentiment",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=5,
    save_strategy="epoch",
    fp16=True,
    learning_rate=3e-4
)

# Step 7: 构建Trainer对象
trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=tokenized_dataset
)

# Step 8: 启动训练流程
trainer.train()

# Step 9: 保存训练好的LoRA模型权重
model.save_pretrained("./output/qwen_lora_sentiment")




# 【例3-4】
import torch
import time
import threading
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer

# Step 1: 自动检测GPU设备
device_ids = [i for i in range(torch.cuda.device_count())]
assert len(device_ids) >= 1, "需要至少一个可用GPU"

# Step 2: 为每张GPU加载一个模型副本
models = []
tokenizers = []

for device_id in device_ids:
    print(f"正在初始化 GPU-{device_id} 上的模型副本...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-1.8B", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen-1.8B",
        trust_remote_code=True,
        torch_dtype=torch.float16
    ).to(f"cuda:{device_id}").eval()
    models.append(model)
    tokenizers.append(tokenizer)

print("所有模型副本加载完成。")

# Step 3: 定义异步推理函数
def run_inference(device_index, input_text):
    tokenizer = tokenizers[device_index]
    model = models[device_index]

    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(f"cuda:{device_index}")
    
    with torch.no_grad():
        start = time.time()
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=128,
            do_sample=True,
            temperature=0.7
        )
        end = time.time()

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)
    print(f"[GPU-{device_index}] 推理耗时：{round(end - start, 2)}秒")
    print(f"[GPU-{device_index}] 输出结果：{decoded}\n")

    # 清理显存
    del input_ids, output
    torch.cuda.empty_cache()

# Step 4: 构造并发请求样本
prompts = [
    "请简述Transformer模型的核心机制。",
    "什么是多模态大模型？其应用场景有哪些？",
    "如何通过LoRA对大语言模型进行微调？",
    "当前AI Agent的主要技术难点有哪些？"
]

# Step 5: 多线程启动推理任务
threads = []
for idx, prompt in enumerate(prompts):
    thread = threading.Thread(target=run_inference, args=(idx % len(device_ids), prompt))
    threads.append(thread)
    thread.start()

# 等待所有线程执行完毕
for thread in threads:
    thread.join()




# 【例3-5】
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.responses import StreamingResponse
import torch
import uvicorn
import json
import time

# Step 1: 初始化模型与tokenizer
model_path = "Qwen/Qwen-1.8B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()

# Step 2: 构造FastAPI服务与数据结构
app = FastAPI()

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    temperature: float = 0.7
    max_tokens: int = 256
    stream: bool = False

# Step 3: 构造Prompt拼接函数
def build_prompt(messages):
    history = []
    for msg in messages:
        role = msg.role
        content = msg.content
        if role == "system":
            history.append(f"[系统提示]：{content}")
        elif role == "user":
            history.append(f"[用户]：{content}")
        elif role == "assistant":
            history.append(f"[助手]：{content}")
    prompt = "\n".join(history) + "\n[助手]："
    return prompt

# Step 4: 流式生成器函数
def stream_response(prompt, max_tokens, temperature):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=True
        )
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("[助手]：")[-1].strip()
    for i in range(0, len(response), 10):
        yield json.dumps({"choices": [{"delta": {"content": response[i:i+10]}}]}) + "\n"
        time.sleep(0.05)

# Step 5: 构造主响应接口
@app.post("/v1/chat/completions")
async def chat_completion(req: ChatRequest):
    prompt = build_prompt(req.messages)
    if req.stream:
        return StreamingResponse(stream_response(prompt, req.max_tokens, req.temperature), media_type="text/event-stream")
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                do_sample=True
            )
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("[助手]：")[-1].strip()
        return {"id": "cmpl-local-001", "object": "chat.completion", "choices": [{"message": {"role": "assistant", "content": result}}]}

# Step 6: 启动服务（可通过命令 uvicorn filename:app --reload 启动）
if __name__ == "__main__":
    uvicorn.run("filename:app", host="0.0.0.0", port=8080)






# 请求内容：
{
  "model": "Qwen/Qwen-1.8B",
  "messages": [
    {"role": "system", "content": "你是一名专业助手"},
    {"role": "user", "content": "请解释一下LoRA技术的基本原理"}
  ],
  "temperature": 0.7,
  "max_tokens": 200,
  "stream": false
}




# 【例3-6】
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import uvicorn
import torch
import json

# 模型初始化
model_path = "Qwen/Qwen-1.8B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.float16
).cuda().eval()

# 定义FastAPI服务
app = FastAPI()

# 定义函数结构
function_registry = {
    "get_weather": {
        "description": "获取城市天气",
        "parameters": ["city"]
    },
    "calculate_sum": {
        "description": "计算两个数字的和",
        "parameters": ["a", "b"]
    }
}

# 模拟函数执行
def get_weather(city):
    return f"{city}的当前天气是晴，气温25°C"

def calculate_sum(a, b):
    return f"{a} + {b} = {int(a) + int(b)}"

def dispatch_function(name, args):
    if name == "get_weather":
        return get_weather(args["city"])
    if name == "calculate_sum":
        return calculate_sum(args["a"], args["b"])
    return "未知函数"

# 数据结构定义
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    stream: bool = False

# Prompt构建
def build_prompt(messages):
    history = []
    for m in messages:
        if m.role == "user":
            history.append(f"[用户]：{m.content}")
        elif m.role == "assistant":
            history.append(f"[助手]：{m.content}")
        elif m.role == "system":
            history.append(f"[系统提示]：{m.content}")
    return "\n".join(history) + "\n[助手]："

# 主处理函数
@app.post("/v1/chat/function_call")
async def function_call(req: ChatRequest):
    prompt = build_prompt(req.messages)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=200)
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 提取函数调用请求（模拟）
    if "调用函数" in output_text:
        if "get_weather" in output_text:
            return {"function_call": {"name": "get_weather", "arguments": {"city": "北京"}}}
        if "calculate_sum" in output_text:
            return {"function_call": {"name": "calculate_sum", "arguments": {"a": "8", "b": "12"}}}
    return {"response": output_text}

# 启动服务命令： uvicorn filename:app --reload
if __name__ == "__main__":
    uvicorn.run("filename:app", host="0.0.0.0", port=8080)





# 请求内容（POST /v1/chat/function_call）：
{
  "model": "Qwen/Qwen-1.8B",
  "messages": [
    {"role": "system", "content": "你可以调用函数获取天气或进行加法"},
    {"role": "user", "content": "请告诉我北京的天气"}
  ],
  "stream": false
}


# 测试结果如下：
{
  "function_call": {
    "name": "get_weather",
    "arguments": {
      "city": "北京"
    }
  }
}




# 【例3-7】
from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from fastapi.responses import StreamingResponse
import torch
import uvicorn
import time
import json

# 初始化模型
model_path = "Qwen/Qwen-1.8B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.float16
).cuda().eval()

# 初始化FastAPI应用
app = FastAPI()
# 输入结构定义
class ChatItem(BaseModel):
    role: str
    content: str

class BatchRequest(BaseModel):
    model: str
    messages_list: list[list[ChatItem]]  # 多组对话历史
    max_tokens: int = 128

class StreamRequest(BaseModel):
    model: str
    messages: list[ChatItem]
    max_tokens: int = 128
    temperature: float = 0.7

# 构造Prompt函数
def build_prompt(messages):
    prompt = ""
    for msg in messages:
        if msg.role == "user":
            prompt += f"[用户]：{msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"[助手]：{msg.content}\n"
        elif msg.role == "system":
            prompt += f"[系统提示]：{msg.content}\n"
    prompt += "[助手]："
    return prompt

# 批处理接口
@app.post("/v1/chat/batch")
async def batch_chat(req: BatchRequest):
    prompts = [build_prompt(messages) for messages in req.messages_list]
    input_ids = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).input_ids.cuda()
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=req.max_tokens)
    decoded = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    results = [text.split("[助手]：")[-1].strip() for text in decoded]
    return {"results": results}

# 流式输出生成器
def stream_generator(prompt, max_tokens, temperature):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature
        )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).split("[助手]：")[-1].strip()
    for i in range(0, len(output_text), 8):
        yield json.dumps({"choices": [{"delta": {"content": output_text[i:i+8]}}]}) + "\n"
        time.sleep(0.05)

# 流式输出接口
@app.post("/v1/chat/stream")
async def stream_chat(req: StreamRequest):
    prompt = build_prompt(req.messages)
    return StreamingResponse(stream_generator(prompt, req.max_tokens, req.temperature), media_type="text/event-stream")

# 启动服务
if __name__ == "__main__":
    uvicorn.run("filename:app", host="0.0.0.0", port=8000)






# 批处理请求：
{
  "model": "Qwen/Qwen-1.8B",
  "messages_list": [
    [{"role": "user", "content": "介绍一下LangChain"}],
    [{"role": "user", "content": "什么是LoRA技术？"}]
  ]
}
# 批处理输出：
{
  "results": [
    "LangChain是一个用于构建基于大模型的链式思维系统的开发框架...",
    "LoRA是一种轻量化微调方法，通过注入低秩矩阵进行模型参数适配..."
  ]
}
# 流失输出请求：
{
  "model": "Qwen/Qwen-1.8B",
  "messages": [
    {"role": "user", "content": "请解释一下大语言模型的上下文窗口"}
  ],
  "max_tokens": 100,
  "temperature": 0.7
}
# 流式响应片段（持续推送）：
{"choices":[{"delta":{"content":"大语言"}}]}
{"choices":[{"delta":{"content":"模型的"}}]}
{"choices":[{"delta":{"content":"上下文窗"}}]}
...




# 【例3-8】
from fastapi import FastAPI, Request
from transformers import AutoTokenizer, AutoModelForCausalLM
from pydantic import BaseModel
import torch
import uvicorn
import json

# 初始化模型
model_path = "Qwen/Qwen-1.8B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()

# 敏感词Trie树结构
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class SensitiveTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, text):
        flagged = []
        for i in range(len(text)):
            node = self.root
            j = i
            while j < len(text) and text[j] in node.children:
                node = node.children[text[j]]
                if node.is_end:
                    flagged.append(text[i:j+1])
                j += 1
        return flagged

# 加载敏感词
sensitive_words = ["暴力", "攻击", "毒品", "敏感政治"]
filter_tree = SensitiveTrie()
for word in sensitive_words:
    filter_tree.insert(word)

# FastAPI初始化
app = FastAPI()

# 定义输入结构
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 128

def build_prompt(messages):
    prompt = ""
    for msg in messages:
        if msg.role == "user":
            prompt += f"[用户]：{msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"[助手]：{msg.content}\n"
        elif msg.role == "system":
            prompt += f"[系统提示]：{msg.content}\n"
    prompt += "[助手]："
    return prompt

@app.post("/v1/chat/audit")
async def chat_filter(req: ChatRequest):
    prompt = build_prompt(req.messages)

    # 生成前过滤
    pre_check = filter_tree.search(prompt)
    if pre_check:
        return {"flag": "input_blocked", "reason": pre_check}

    # 模型推理
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        output_ids = model.generate(input_ids, max_new_tokens=req.max_tokens)
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # 生成后过滤
    clean_text = output_text.split("[助手]：")[-1].strip()
    post_check = filter_tree.search(clean_text)
    if post_check:
        return {"flag": "output_blocked", "reason": post_check, "original": clean_text}

    return {"flag": "ok", "response": clean_text}

# 启动命令： uvicorn filename:app --reload








# 输入内容：
{
  "model": "Qwen/Qwen-1.8B",
  "messages": [
    {"role": "user", "content": "请告诉我关于毒品的知识"}
  ],
  "max_tokens": 64
}
# 输出内容：
{
  "flag": "input_blocked",
  "reason": ["毒品"]
}
# 若内容未命中敏感词：
{
  "flag": "ok",
  "response": "大语言模型是基于Transformer架构构建的序列生成模型..."
}
# 【例3-9】
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F
import uvicorn

# 初始化模型
model_path = "Qwen/Qwen-1.8B"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.float16).cuda().eval()

# FastAPI初始化
app = FastAPI()

# 请求结构定义
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    model: str
    messages: list[Message]
    max_tokens: int = 64
    top_k: int = 5

# 构建Prompt函数
def build_prompt(messages):
    prompt = ""
    for msg in messages:
        if msg.role == "user":
            prompt += f"[用户]：{msg.content}\n"
        elif msg.role == "assistant":
            prompt += f"[助手]：{msg.content}\n"
    prompt += "[助手]："
    return prompt

@app.post("/v1/chat/with_confidence")
async def chat_with_confidence(req: ChatRequest):
    prompt = build_prompt(req.messages)
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        output = model.generate(
            input_ids=input_ids,
            max_new_tokens=req.max_tokens,
            return_dict_in_generate=True,
            output_scores=True
        )
    generated_ids = output.sequences[0]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).split("[助手]：")[-1].strip()

    token_scores = []
    for i, scores in enumerate(output.scores):
        probs = F.softmax(scores[0], dim=-1)
        topk_probs, topk_indices = torch.topk(probs, k=req.top_k)
        tokens = [tokenizer.decode([idx.item()]) for idx in topk_indices]
        probs_float = [float(p) for p in topk_probs]
        token_scores.append({
            "step": i,
            "top_tokens": tokens,
            "top_probs": probs_float
        })

    return {
        "response": generated_text,
        "token_confidence": token_scores
    }

# 启动命令：uvicorn filename:app --reload





# 请求：
{
  "model": "Qwen/Qwen-1.8B",
  "messages": [
    {"role": "user", "content": "请解释一下LangChain的核心组件"}
  ],
  "max_tokens": 32,
  "top_k": 5
}
# 输出：
{
  "response": "LangChain的核心组件包括LLM接口、Chains链式结构、工具集成模块与记忆系统...",
  "token_confidence": [
    {
      "step": 0,
      "top_tokens": ["L", "兰", "数", "能", "工"],
      "top_probs": [0.83, 0.05, 0.03, 0.02, 0.01]
    },
    {
      "step": 1,
      "top_tokens": ["a", "g", "新", "用", "型"],
      "top_probs": [0.76, 0.08, 0.05, 0.03, 0.01]
    },
    ...
  ]
}
