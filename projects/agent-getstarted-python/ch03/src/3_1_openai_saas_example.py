# 【例3-1】
# openai_saas_example.py

import sys
# import requests  # 可选：如果需要直接调用HTTP接口
from pathlib import Path
from typing import List, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import chat_completion

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
        response = chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        reply = response.choices[0].message.content
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
