# 【例4-1】
# 4_1_research_chain.py
#
# 本例演示一个三步研究选题链：
# 1. 根据主题生成细分研究方向
# 2. 为每个方向生成研究问题
# 3. 总结每个研究问题的社会价值
#
# 本地Ollama推荐运行：
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch04/src/4_1_research_chain.py
#
# 云端DeepSeek/OpenAI兼容API运行：
# python3 ch04/src/4_1_research_chain.py

import json
from typing import Any

from langchain_core.prompts import PromptTemplate

from llm_backend import ask_json, ask_text, backend_name

SYSTEM_PROMPT = (
    "你是一名严谨的中文科研选题助手。"
    "回答必须简洁、结构化，不要输出思考过程，不要输出Markdown。"
    "所有JSON字符串都必须使用英文双引号并正确闭合。"
)


def invoke_json(prompt: str, temperature: float = 0, max_tokens: int = 600) -> dict[str, Any]:
    return ask_json(
        prompt,
        system=SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=max_tokens,
    )


prompt1 = PromptTemplate(
    input_variables=["topic"],
    template=(
        "主题：{topic}\n"
        "请生成3个细分研究方向。\n"
        "只输出JSON对象，格式如下：\n"
        '{{"subtopics":["方向1","方向2","方向3"]}}\n'
        "要求：每个方向不超过12个中文字，不要解释。"
    ),
)

prompt2 = PromptTemplate(
    input_variables=["subtopics"],
    template=(
        "细分研究方向：{subtopics}\n"
        "请为每个方向提出1个具挑战性的研究问题。\n"
        "只输出JSON对象，格式如下：\n"
        '{{"questions":[{{"subtopic":"方向1","question":"问题1"}}]}}\n'
        "要求：每个问题不超过70个中文字，不要写分析。"
    ),
)

def ensure_list(data: dict[str, Any], key: str) -> list[Any]:
    value = data.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"模型输出缺少有效字段：{key}，原始输出：{data}")
    return value


def summarize_impact(subtopic: str, question: str) -> str:
    return ask_text(
        f"细分方向：{subtopic}\n研究问题：{question}\n请总结这个研究问题的社会价值。",
        system=SYSTEM_PROMPT + "只输出一句中文，不超过45个中文字。",
        temperature=0,
        max_tokens=120,
    ).strip()


def run_research_chain(topic: str) -> dict[str, Any]:
    subtopic_data = invoke_json(prompt1.format(topic=topic), max_tokens=256)
    subtopics = ensure_list(subtopic_data, "subtopics")[:3]

    question_data = invoke_json(
        prompt2.format(subtopics=json.dumps(subtopics, ensure_ascii=False)),
        max_tokens=512,
    )
    questions = ensure_list(question_data, "questions")[:3]

    impacts = [
        {
            "subtopic": item.get("subtopic", "未命名方向"),
            "impact": summarize_impact(
                item.get("subtopic", "未命名方向"),
                item.get("question", ""),
            ),
        }
        for item in questions
    ]

    return {
        "topic": topic,
        "backend": backend_name(),
        "subtopics": subtopics,
        "questions": questions,
        "impacts": impacts,
    }


def print_research_chain(result: dict[str, Any]):
    print(f"主题：{result['topic']}")
    print(f"LLM后端：{result['backend']}")

    print("\n细分研究方向：")
    for index, subtopic in enumerate(result["subtopics"], start=1):
        print(f"{index}. {subtopic}")

    print("\n研究问题：")
    for index, item in enumerate(result["questions"], start=1):
        print(f"{index}. [{item.get('subtopic', '未命名方向')}] {item.get('question', '')}")

    print("\n社会价值：")
    for index, item in enumerate(result["impacts"], start=1):
        print(f"{index}. [{item.get('subtopic', '未命名方向')}] {item.get('impact', '')}")


if __name__ == "__main__":
    response = run_research_chain("人工智能伦理")
    print_research_chain(response)
