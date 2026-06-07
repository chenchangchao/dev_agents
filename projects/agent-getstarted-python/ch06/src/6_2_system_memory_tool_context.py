# 【例6-2】系统段、记忆段、工具段与最终生成
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch06/src/6_2_system_memory_tool_context.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch06/src/6_2_system_memory_tool_context.py

import json

from ch06_runtime import Tool, ask_llm, backend_name, build_segment, format_segments


class FiscalForecastTool(Tool):
    def __init__(self):
        super().__init__(name="fiscal_forecast", description="财政收入趋势预测工具")

    def call(self, query: str) -> str:
        prompt = f"请对以下财政问题给出结构化预测，只输出关键判断：{query}"
        return ask_llm(
            prompt,
            system="你是财政数据预测工具，回答要简短、偏结构化。",
            max_tokens=260,
            label="fiscal_forecast_tool",
        )


def build_segments(tool: FiscalForecastTool) -> list[dict]:
    task_id = "t002"
    segments = [
        build_segment(
            "system",
            "你是一位熟悉中国经济政策的政府咨询顾问，回答需专业、简明、有依据。",
            "system",
            task_id,
        ),
        build_segment("memory", "用户偏好获取简明扼要的政策摘要，避免冗长表述。", "memory", task_id),
    ]
    user_question = "请预测2024年中国的财政收入趋势，并说明主要增长来源。"
    segments.append(build_segment("user", user_question, "input", task_id))

    tool_response = tool.call(user_question)
    segments.append(
        build_segment(
            "tool",
            tool_response,
            "tool",
            task_id,
            model=backend_name(),
            meta={"tool": tool.name},
        )
    )
    return segments


def generate_final_answer(segments: list[dict]) -> str:
    final_prompt = f"""以下是结构化上下文段，请基于这些内容给出准确回答：

{format_segments(segments)}

请输出：1. 趋势判断；2. 主要增长来源；3. 风险提示。"""
    return ask_llm(
        final_prompt,
        system="你是一位财政政策问答助手，只能基于给定上下文回答。",
        max_tokens=420,
        label="final_answer",
    )


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    tool = FiscalForecastTool()
    segments = build_segments(tool)
    response = generate_final_answer(segments)

    print("\n--- Agent 最终回答 ---")
    print(response)

    print("\n--- 当前上下文段结构 ---")
    print(json.dumps(segments, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
