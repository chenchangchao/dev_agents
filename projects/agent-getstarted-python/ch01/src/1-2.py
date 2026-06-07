import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import chat_completion

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

result = chat_completion(messages=[{"role": "user", "content": prompt}])

print(result.choices[0].message.content)


# 以下是典型的接口模式设计：
# 构建消息序列
messages = [
    {"role": "system", "content": "你是一位金融知识专家，请严格按照要求回答问题。"},
    {"role": "user", "content": "请告诉我债券和股票的区别，用表格列出。"},
]

# 调用模型
response = chat_completion(messages=messages, temperature=0.3, max_tokens=512)

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

response = chat_completion(
    temperature=0.3,
    max_tokens=512,
    messages=[{"role": "user", "content": prompt}],
)

print(response.choices[0].message.content)
