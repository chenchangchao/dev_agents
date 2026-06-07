# 【例6-7】动态上下文合并策略
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch06/src/6_7_dynamic_context_merge.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch06/src/6_7_dynamic_context_merge.py

from datetime import datetime

from ch06_runtime import Tool, ask_llm, backend_name, build_segment


class MacroForecastTool(Tool):
    def __init__(self):
        super().__init__(name="macro_forecast", description="经济趋势分析工具")

    def call(self, query: str) -> str:
        print("→ 宏观预测工具调用中...")
        return ask_llm(
            f"请针对以下问题输出简要宏观预测：{query}",
            system="你是宏观预测工具，回答要短、可作为上下文证据。",
            max_tokens=260,
            label="macro_forecast",
        )


def timestamp_value(segment: dict) -> str:
    return segment.get("timestamp") or datetime.now().isoformat()


def dynamic_merge_context(segments: list[dict], task_type: str) -> str:
    selected = []
    selected.extend([item for item in segments if item["type"] == "system"])

    tool_segments = sorted(
        [item for item in segments if item["type"] == "tool"],
        key=timestamp_value,
        reverse=True,
    )
    selected.extend(tool_segments[:1])

    history = sorted(
        [item for item in segments if item["type"] in ["input", "response"]],
        key=timestamp_value,
        reverse=True,
    )
    selected.extend(history[:4])

    lines = [f"任务类型：{task_type}"]
    for seg in selected:
        prefix = f"[{seg['role'].upper()}-{seg['type']}]"
        lines.append(f"{prefix} {seg['content']}")
    return "\n".join(lines)


def build_context_pool(tool: MacroForecastTool) -> list[dict]:
    task_id = "t007"
    context_pool = [
        build_segment(
            "system",
            "你是一位国家政策与宏观经济专家，语言要严谨、简洁、具数据支撑。",
            "system",
            task_id,
        ),
        build_segment("user", "2022年GDP增长率是多少？", "input", task_id),
        build_segment("assistant", "2022年中国GDP增长为3.0%。", "response", task_id, model="example"),
        build_segment("user", "2023年预计呢？", "input", task_id),
        build_segment("assistant", "2023年GDP增长约为5.2%。", "response", task_id, model="example"),
    ]

    tool_output = tool.call("2024年GDP预测")
    context_pool.append(build_segment("tool", tool_output, "tool", task_id, model=backend_name(), meta={"tool": tool.name}))
    context_pool.append(build_segment("user", "请基于历史数据与趋势，判断2024年GDP是否可能超过5.5%？", "input", task_id))
    return context_pool


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动动态上下文合并策略系统 ===")

    tool = MacroForecastTool()
    context_pool = build_context_pool(tool)
    final_prompt = dynamic_merge_context(context_pool, task_type="forecast")

    print("\n--- 合并后的Prompt结构 ---")
    print(final_prompt)

    answer = ask_llm(
        final_prompt,
        system="你是一位宏观经济分析助手，只能基于给定上下文做审慎判断。",
        max_tokens=420,
        label="dynamic_context",
    )
    print("\n--- 模型输出结果 ---")
    print(answer)


if __name__ == "__main__":
    main()
