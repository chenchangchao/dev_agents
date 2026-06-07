# 【例12-8】主控 Agent 集成：多工具 + 记忆链
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch12/src/12_8_integrated_agent_tools_memory.py

from ch12_runtime import ask_llm, backend_name, extract_city, extract_numbers, safe_multiply


def get_temperature(city: str) -> str:
    if city == "未知" or city == "月球":
        raise ValueError("未找到对应城市")
    return f"{city}当前温度为26摄氏度。"


class IntegratedAgent:
    def __init__(self):
        self.memory: list[tuple[str, str]] = []

    def run(self, input_text: str) -> str:
        try:
            if "温度" in input_text or "天气" in input_text:
                result = get_temperature(extract_city(input_text))
            else:
                nums = extract_numbers(input_text)
                if len(nums) >= 2 and ("乘" in input_text or "*" in input_text):
                    result = safe_multiply(nums[0], nums[1])
                else:
                    result = ask_llm(f"历史对话：{self.memory}\n用户输入：{input_text}", temperature=0.3, max_tokens=300, label="integrated")
            self.memory.append((input_text, result))
            return result
        except Exception:
            return "调用失败，请稍后重试。"


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    agent = IntegratedAgent()
    for index, text in enumerate(["请帮我查询北京的当前温度", "将7和8相乘是多少", "请查一下月球的天气", "再帮我乘一下12和13"], start=1):
        print(f"\n输入{index}：{text}")
        print("输出结果：", agent.run(text))


if __name__ == "__main__":
    main()
