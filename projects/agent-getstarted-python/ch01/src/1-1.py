# 案例一：生成SQL语句的任务建模
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import chat_completion

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
response = chat_completion(messages=[{"role": "user", "content": prompt}], temperature=0)

# 输出结果
print(response.choices[0].message.content)
