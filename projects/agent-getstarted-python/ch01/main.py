# 第1章 基于智能体的大模型开发模式简介
# 以下是一个用于文本摘要的提示词设计示例：
from openai import OpenAI 
import dotenv  # 加载环境变量

dotenv.load_dotenv()  # 从.env文件加载环境变量
import os

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_MODEL_ID = os.getenv("DEEPSEEK_MODEL_ID")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL")

prompt = """
你是一位专业的中文技术文档撰写专家，
请将以下文本进行简洁、准确的摘要，
要求语言通顺，保留技术要点，控制在100字以内：

【原文内容】：
{}
"""

document = "Transformer结构是一种基于注意力机制的深度学习架构..."

# 拼接输入内容
final_prompt = prompt.format(document)

# 调用模型生成
response = OpenAI(
    api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL
).chat.completions.create(
    model=DEEPSEEK_MODEL_ID,
    messages=[{"role": "user", "content": final_prompt}],
    temperature=0.7,
)

print(response.choices[0].message.content)


# 以下示例展示如何使用LangChain的提示词模板构建任务输入：
from langchain_core.prompts import PromptTemplate

# 定义提示词模板
template = PromptTemplate(
    input_variables=["question"],
    template="""
你是一位知识图谱专家，请基于已有知识，回答以下问题，
并说明判断依据：

问题：{question}
答案：
""",
)

# 执行任务
final_prompt = template.format(question="什么是实体消歧？")
response = OpenAI(
    api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL
).chat.completions.create(
    model=DEEPSEEK_MODEL_ID,
    messages=[{"role": "user", "content": final_prompt}],
    temperature=0.7,
)
print(response.choices[0].message.content)


# 案例一：生成SQL语句的任务建模
from openai import OpenAI  # 使用OpenAI兼容接口

llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 构建任务Prompt
user_instruction = "我想查询所有注册时间在2023年之后的用户姓名和邮箱"

prompt = f"""
你是一位数据库专家，请根据以下用户指令生成SQL语句。
要求：
1. 表名为users；
2. 字段包括name和email；
3. 使用标准SQL语法；
4. 输出格式为JSON，字段包括sql和explanation。

指令：{user_instruction}
"""

# 调用大模型
response = llm.chat.completions.create(
    model=DEEPSEEK_MODEL_ID, messages=[{"role": "user", "content": prompt}], temperature=0
)

# 输出结果
print(response.choices[0].message.content)


# 案例二：文本抽取任务建模（抽取企业名与地址）
instruction = "从以下文本中提取公司名称和地址信息，以JSON格式返回。\n\n文本：浙江橙龙科技有限公司位于杭州市余杭区五常街道五常大道100号。"

prompt = f"""
任务说明：从文本中识别并提取公司名称与地址信息。

输出格式：
{{
  "company": "...",
  "address": "..."
}}

请处理如下文本：
{instruction}
"""

result = llm.chat.completions.create(
    model=DEEPSEEK_MODEL_ID, messages=[{"role": "user", "content": prompt}]
)

print(result.choices[0].message.content)


# 以下是典型的接口模式设计：
from openai import OpenAI

llm = OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)

# 构建消息序列
messages = [
    {"role": "system", "content": "你是一位金融知识专家，请严格按照要求回答问题。"},
    {"role": "user", "content": "请告诉我债券和股票的区别，用表格列出。"},
]

# 调用模型
response = llm.chat.completions.create(
    model=DEEPSEEK_MODEL_ID, messages=messages, temperature=0.3, max_tokens=512
)

print(response.choices[0].message.content)


# 以下是一个简单示例，展示如何通过上下文信息引导大模型分阶段完成任务：
prompt = """
你是一位智能体助手，当前任务是多步骤问题求解。

任务目标：用户希望了解其信用卡的账单日、还款日，并获取本月账单详情。

已知上下文：
- 用户身份验证已完成；
- 已完成步骤：查询账单日，返回为每月15日；
- 当前阶段：继续查询还款日。

请根据上述上下文，生成用户当前还需了解的信息，并指导下一步行为。
"""

response = llm.chat.completions.create(
    model=DEEPSEEK_MODEL_ID,
    temperature=0.3,
    max_tokens=512,
    messages=[{"role": "user", "content": prompt}],
)

print(response.choices[0].message.content)
