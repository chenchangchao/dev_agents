# 【例6-1】结构化上下文段与双模型视角融合
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch06/src/6_1_structured_context_fusion.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch06/src/6_1_structured_context_fusion.py

import json

from ch06_runtime import Tool, ask_llm, backend_name, build_segment, format_segments


class StructuredContextQueryTool(Tool):
    def __init__(self, segments: list[dict]):
        super().__init__(
            name="structured_context_query",
            description="结构化上下文段解读工具",
        )
        self.segments = segments

    def call(self, query: str) -> str:
        relevant = [seg for seg in self.segments if seg["type"] in ["input", "response"]]
        context_summary = format_segments(relevant, limit=4)
        return f"【对话上下文】\n{context_summary}\n\n【用户问题】\n{query}"


def build_demo_segments() -> list[dict]:
    task_id = "t001"
    return [
        build_segment("user", "我想了解2023年中国的财政支出情况。", "input", task_id),
        build_segment(
            "assistant",
            "2023年中国财政支出约为26万亿元，主要用于民生保障、教育和基础设施建设。",
            "response",
            task_id,
            model="example",
        ),
        build_segment("user", "那2022年与2023年相比增长了多少？", "input", task_id),
    ]


def generate_policy_view(prompt: str) -> str:
    return ask_llm(
        prompt,
        system="你是一位财政政策分析助手，回答要专业、简洁、注意上下文一致性。",
        max_tokens=360,
        label="policy_view",
    )


def generate_data_view(prompt: str) -> str:
    return ask_llm(
        prompt,
        system="你是一位结构化数据分析助手，请从趋势、同比、风险角度补充判断。",
        max_tokens=360,
        label="data_view",
    )


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    context_segments = build_demo_segments()
    context_tool = StructuredContextQueryTool(context_segments)

    query = "是否存在财政支出快速增长的趋势？"
    structured_prompt = context_tool.call(query)

    policy_response = generate_policy_view(structured_prompt)
    data_response = generate_data_view(structured_prompt)

    print("\n--- 政策视角回答 ---")
    print(policy_response)
    print("\n--- 数据视角辅助回答 ---")
    print(data_response)

    context_segments.append(build_segment("assistant", policy_response, "response", "t001", model=backend_name()))
    context_segments.append(
        build_segment(
            "tool",
            data_response,
            "observation",
            "t001",
            model=backend_name(),
            meta={"source": "data_view"},
        )
    )

    print("\n--- 当前上下文段结构 ---")
    print(json.dumps(context_segments, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
