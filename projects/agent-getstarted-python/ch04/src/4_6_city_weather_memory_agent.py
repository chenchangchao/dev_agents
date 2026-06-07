# 【例4-6】
# 4_6_city_weather_memory_agent.py
#
# 本例演示外部HTTP工具与对话记忆：
# 1. wttr.in 查询城市天气
# 2. Wikipedia API 获取城市简介
# 3. LLM将英文简介改写为中文摘要
#
# 本地Ollama运行：
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch04/src/4_6_city_weather_memory_agent.py
#
# 云端DeepSeek/OpenAI兼容API运行：
# python3 ch04/src/4_6_city_weather_memory_agent.py

import requests

from llm_backend import ask_text, backend_name


# 工具函数：天气查询
def get_weather(city: str) -> str:
    try:
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url, timeout=10)
        return response.text.strip()
    except Exception as e:
        return f"查询失败：{str(e)}"


# 工具函数：城市百科简介（调用维基百科API）
def get_city_intro(city: str) -> str:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{city}"
        response = requests.get(url, timeout=10).json()
        return response.get("extract", "未找到该城市简介")
    except Exception as e:
        return f"查询失败：{str(e)}"


class ConversationMemory:
    def __init__(self):
        self.messages = []

    def add(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def text(self) -> str:
        return "\n".join(f"{item['role']}: {item['content']}" for item in self.messages)


def detect_city(user_input: str) -> str | None:
    for city in ["上海", "北京", "Guangzhou", "Shanghai", "Beijing"]:
        if city in user_input:
            return city
    return None


def run_agent(user_input: str, memory: ConversationMemory) -> str:
    city = detect_city(user_input)
    if city and "天气" in user_input:
        answer = get_weather(city)
    elif city and ("介绍" in user_input or "简介" in user_input):
        intro = get_city_intro(city)
        answer = ask_text(
            f"请把下面城市介绍用中文简要改写：\n{intro}",
            system="你是一个城市旅行助手。回答不超过160字，不要输出思考过程。",
            temperature=0.5,
            max_tokens=300,
        )
    else:
        answer = ask_text(
            f"历史对话：\n{memory.text()}\n\n用户问题：{user_input}",
            system="你是一位知识渊博的智能助理，请结合历史和可用工具结果回答。",
            temperature=0.5,
            max_tokens=300,
        )

    memory.add("user", user_input)
    memory.add("assistant", answer)
    return answer


def main():
    print(f"LLM后端：{backend_name()}\n")
    memory = ConversationMemory()

    # 启动Agent并模拟多轮对话
    print("====== 实际对话示例 ======")
    print(run_agent("介绍一下上海", memory))
    print(run_agent("那现在上海的天气如何？", memory))
    print(run_agent("北京天气怎么样？", memory))


if __name__ == "__main__":
    main()
