# 【例7-2】自然语言输入参数解析与封装
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch07/src/7_2_input_parameter_parser.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch07/src/7_2_input_parameter_parser.py

import re

from ch07_runtime import Tool, ask_llm, backend_name


class EconomicForecastTool(Tool):
    def __init__(self):
        super().__init__(name="economic_forecast", description="基于区域和时间预测经济增长")

    def call(self, params: dict) -> str:
        region = params.get("region", "中国")
        year = params.get("year", 2024)
        focus = params.get("focus", "GDP")
        return f"[Tool] 预测结果：{region}在{year}年的{focus}增长趋势预计保持温和修复。"


def parse_input(user_text: str) -> dict:
    year_match = re.search(r"(20\d{2})年?", user_text)
    region = "中国"
    if "美国" in user_text:
        region = "美国"
    elif "欧盟" in user_text:
        region = "欧盟"

    focus = "GDP"
    if "CPI" in user_text.upper():
        focus = "CPI"
    elif "PMI" in user_text.upper():
        focus = "PMI"

    return {"region": region, "year": int(year_match.group(1)) if year_match else 2024, "focus": focus}


def print_parameters(params: dict) -> None:
    print("\n--- 结构化参数 ---")
    for key, value in params.items():
        print(f"{key}: {value}")


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动输入参数解析与封装系统 ===")
    tool = EconomicForecastTool()
    user_input = "请预测2024年美国的GDP增长趋势"

    structured_params = parse_input(user_input)
    print_parameters(structured_params)

    tool_output = tool.call(structured_params)
    print("\n--- 工具输出结果 ---")
    print(tool_output)

    prompt = f"""请根据以下工具数据生成简洁经济解读：

工具输出：{tool_output}
问题原文：{user_input}

请用政策分析师的视角进行总结。"""
    answer = ask_llm(prompt, system="你是一位经济政策分析师。", max_tokens=360, label="parameter_parser")
    print("\n--- 模型输出结果 ---")
    print(answer)


if __name__ == "__main__":
    main()
