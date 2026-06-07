"""例12-9：工具调用正确率评估。

运行方式：
cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
python3 ch12/src/12_9_tool_call_accuracy_eval.py
LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch12/src/12_9_tool_call_accuracy_eval.py
"""

from __future__ import annotations

from dataclasses import dataclass

from ch12_runtime import backend_name, extract_city, extract_numbers, safe_multiply


def weather_tool(city: str) -> str:
    if city not in ["北京", "上海", "广州", "深圳"]:
        raise ValueError("城市不支持")
    return f"weather_tool: {city}天气为晴，温度25°C"


def multiply_tool(a: str, b: str) -> str:
    return f"multiply_tool: {safe_multiply(a, b)}"


@dataclass
class ToolCase:
    text: str
    expected_tool: str


class ToolTestAgent:
    def run(self, prompt: str) -> str:
        try:
            city = extract_city(prompt)
            if city != "未知" and any(word in prompt for word in ["天气", "气温", "温度"]):
                return weather_tool(city)

            nums = extract_numbers(prompt)
            if len(nums) >= 2 and any(word in prompt for word in ["乘", "*", "相乘", "计算"]):
                return multiply_tool(nums[0], nums[1])

            return "未匹配到合适工具"
        except Exception as exc:
            return f"[ERROR] {exc}"


TEST_CASES = [
    ToolCase("请查询一下北京的天气", "weather_tool"),
    ToolCase("上海今天气温多少？", "weather_tool"),
    ToolCase("广州天气怎么样？", "weather_tool"),
    ToolCase("请帮我计算7乘以8", "multiply_tool"),
    ToolCase("12和13相乘是多少？", "multiply_tool"),
    ToolCase("你知道深圳温度吗？", "weather_tool"),
]


def run_tool_test() -> float:
    agent = ToolTestAgent()
    correct = 0

    print(f"后端：{backend_name()} | 本例评估工具路由规则，不强制调用LLM")
    print(f"总用例数：{len(TEST_CASES)}")

    for index, case in enumerate(TEST_CASES, start=1):
        output = agent.run(case.text)
        matched = case.expected_tool in output
        correct += int(matched)
        mark = "通过" if matched else "失败"
        print(f"{index}. {mark} | 输入：{case.text}")
        print(f"   输出：{output}")

    accuracy = correct / len(TEST_CASES)
    print(f"\n工具调用正确率：{correct}/{len(TEST_CASES)} = {accuracy:.2f}")
    return accuracy


def main() -> None:
    run_tool_test()


if __name__ == "__main__":
    main()
