# 【例12-3】工具链调用与回退机制
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch12/src/12_3_tool_chain_fallback.py

import re
import traceback

from ch12_runtime import ask_llm, backend_name, extract_city


def query_weather(city: str) -> str:
    if city == "未知":
        raise ValueError("无法获取指定城市天气信息")
    return f"{city}当前天气：晴，气温26摄氏度。"


def calculate_expression(expr: str) -> str:
    try:
        result = eval(expr, {"__builtins__": {}})
        return f"计算结果为：{result}"
    except Exception:
        return "表达式有误，无法计算。"


class ToolAgentManager:
    def run_with_tools(self, input_text: str) -> str:
        try:
            if "天气" in input_text:
                return query_weather(extract_city(input_text))
            expr_match = re.search(r"[\d+\-*/(). ]+", input_text)
            if "计算" in input_text and expr_match:
                return calculate_expression(expr_match.group(0).strip())
            return ask_llm(input_text, temperature=0.3, max_tokens=300, label="tool_fallback")
        except Exception:
            print("[WARN] 工具执行失败，回退至模型直接回答")
            print(traceback.format_exc())
            return ask_llm(input_text, temperature=0.3, max_tokens=300, label="tool_fallback")


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    manager = ToolAgentManager()
    inputs = ["帮我查一下北京的天气", "计算 7 * (8 + 3)", "请查一下未知城市的天气", "将这段话翻译为英文：你好"]
    for index, text in enumerate(inputs, start=1):
        print(f"\n输入{index}：{text}")
        print("输出结果：", manager.run_with_tools(text))


if __name__ == "__main__":
    main()
