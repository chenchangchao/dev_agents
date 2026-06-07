# 以下是一个用于文本摘要的提示词设计示例：
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import chat_completion

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
response = chat_completion(
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
response = chat_completion(
    messages=[{"role": "user", "content": final_prompt}],
    temperature=0.7,
)
print(response.choices[0].message.content)
