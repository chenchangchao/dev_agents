# 【例7-5】动态上下文剪辑
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch07/src/7_5_dynamic_context_clipping.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch07/src/7_5_dynamic_context_clipping.py

from ch07_runtime import Tool, ask_llm, backend_name, build_segment


class EconomicSummaryTool(Tool):
    def __init__(self):
        super().__init__(name="economic_summary", description="生成结构化经济总结")

    def call(self, query: str) -> str:
        return f"[Tool] 分析完成：{query} 的GDP趋势整体偏稳，政策支持仍是重要变量。"


def dynamic_clip_context(segments: list[dict], token_limit: int = 150) -> str:
    sorted_segments = sorted(segments, key=lambda item: (-item["priority"], -item["timestamp"]))
    total = 0
    clipped = []
    for segment in sorted_segments:
        if total + segment["length"] <= token_limit:
            clipped.append(segment)
            total += segment["length"]

    return "\n".join(f"[{seg['role']}-{seg['type']}] {seg['content']}" for seg in clipped)


def build_context_pool(tool: EconomicSummaryTool) -> list[dict]:
    return [
        build_segment("system", "你是资深经济顾问，语言需精准、简洁。", "system"),
        build_segment("user", "2022年GDP是多少？", "input"),
        build_segment("assistant", "2022年增长为3.0%。", "response"),
        build_segment("user", "那2023年呢？", "input"),
        build_segment("assistant", "2023年为5.2%。", "response"),
        build_segment("tool", tool.call("2024年预测"), "tool", priority=0.8),
        build_segment("user", "2024年会不会更高？", "input"),
        build_segment("assistant", "预计增长可达5.4%左右。", "response"),
        build_segment("user", "主要依据有哪些？", "input"),
    ]


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动具备上下文动态剪辑能力的Agent系统 ===")
    tool = EconomicSummaryTool()
    context_pool = build_context_pool(tool)
    prompt = dynamic_clip_context(context_pool, token_limit=100)

    print("\n--- 剪辑后上下文结构 ---")
    print(prompt)

    answer = ask_llm(prompt, system="你是宏观经济问答助手，只能基于剪辑后的上下文回答。", max_tokens=380, label="context_clipping")
    print("\n--- 模型响应输出 ---")
    print(answer)


if __name__ == "__main__":
    main()
