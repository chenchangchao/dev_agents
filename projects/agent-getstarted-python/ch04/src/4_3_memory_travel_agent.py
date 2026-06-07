# 【例4-3】
# 4_3_memory_travel_agent.py
#
# 本例演示带短期记忆的旅行建议Agent：
# 1. 使用ConversationBufferMemory保存多轮对话
# 2. 天气问题优先走本地模拟天气工具
# 3. 旅行建议交给LLM结合历史回答
#
# 本地Ollama运行：
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch04/src/4_3_memory_travel_agent.py
#
# 云端DeepSeek/OpenAI兼容API运行：
# python3 ch04/src/4_3_memory_travel_agent.py

import re
from dataclasses import dataclass, field

from llm_backend import ask_text, backend_name


@dataclass
class ConversationBufferMemory:
    messages: list[dict[str, str]] = field(default_factory=list)

    def add_user_message(self, content: str):
        self.messages.append({"role": "user", "content": content})

    def add_ai_message(self, content: str):
        self.messages.append({"role": "assistant", "content": content})

    def history_text(self) -> str:
        role_names = {"user": "用户", "assistant": "助手"}
        return "\n".join(
            f"{role_names.get(message['role'], message['role'])}：{message['content']}"
            for message in self.messages
        )


# 自定义工具：天气响应模拟
def get_weather(city: str) -> str:
    dummy_weather = {
        "北京": "晴，25°C",
        "上海": "多云，23°C",
        "广州": "雷阵雨，29°C"
    }
    return dummy_weather.get(city, "当前城市暂无天气数据")


def ask_llm(user_input: str, memory: ConversationBufferMemory) -> str:
    return ask_text(
        f"对话历史：\n{memory.history_text()}\n\n当前问题：{user_input}",
        system=(
            "你是一个旅行建议助手。请参考对话历史回答用户问题。"
            "如果历史中出现城市天气，请结合天气给出简短建议。"
            "回答不超过120字，不要输出思考过程。"
        ),
        temperature=0.4,
        max_tokens=512,
    )


def run_agent(user_input: str, memory: ConversationBufferMemory) -> str:
    city_match = re.search(r"(北京|上海|广州)", user_input)
    if city_match and ("天气" in user_input or "那" in user_input):
        city = city_match.group(1)
        answer = f"{city}天气：{get_weather(city)}"
    else:
        answer = ask_llm(user_input, memory)

    memory.add_user_message(user_input)
    memory.add_ai_message(answer)
    return answer

def main():
    print(f"LLM后端：{backend_name()}\n")
    memory = ConversationBufferMemory()

    # 连续执行多轮对话任务
    print("用户：我想知道北京天气如何？")
    res1 = run_agent("我想知道北京天气如何？", memory)
    print("回答：", res1)

    print("\n用户：那广州呢？")
    res2 = run_agent("那广州呢？", memory)
    print("回答：", res2)

    print("\n用户：那明天去哪旅游好？")
    res3 = run_agent("那明天去哪旅游好？", memory)
    print("回答：", res3)


if __name__ == "__main__":
    main()
