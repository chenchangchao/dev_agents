# 【例6-4】模型入口决策与混合分析
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch06/src/6_4_model_entry_router.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch06/src/6_4_model_entry_router.py

from ch06_runtime import Tool, ask_llm, backend_name


def call_policy_model(prompt: str) -> str:
    print("→ 政策模型入口已触发。")
    return ask_llm(
        prompt,
        system="你是政策解读助手，请给出简洁、准确的政策解释。",
        max_tokens=300,
        label="policy_model",
    )


def call_data_model(prompt: str) -> str:
    print("→ 数据模型入口已触发。")
    return ask_llm(
        prompt,
        system="你是经济预测助手，请给出趋势判断和不确定性说明。",
        max_tokens=300,
        label="data_model",
    )


def route_decision(query: str) -> str:
    if any(keyword in query for keyword in ["综合分析", "多维度", "对比"]):
        return "hybrid"
    if any(keyword in query for keyword in ["政策", "法规", "概述", "总结"]):
        return "policy"
    if any(keyword in query for keyword in ["预测", "趋势", "估计", "数据"]):
        return "data"
    return "default"


class HybridAnalysisTool(Tool):
    def __init__(self):
        super().__init__(name="hybrid_analysis", description="多模型结果融合工具")

    def call(self, inputs: dict[str, str]) -> str:
        policy_result = call_policy_model(inputs["policy"])
        data_result = call_data_model(inputs["forecast"])
        return f"[组合分析]\n政策视角：{policy_result}\n\n数据预测视角：{data_result}"


def handle_query(query: str, tool: HybridAnalysisTool) -> str:
    decision = route_decision(query)
    if decision == "policy":
        return call_policy_model(query)
    if decision == "data":
        return call_data_model(query)
    if decision == "hybrid":
        return tool.call({"policy": "财政补贴政策分析", "forecast": "2024年经济趋势"})
    return ask_llm(query, system="你是默认问答助手。", max_tokens=260, label="default")


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动模型入口决策示例系统 ===")

    user_queries = [
        "请概述近期财政补贴政策的主要方向",
        "请预测2024年GDP增长趋势",
        "请对近期财政补贴政策与经济增长趋势进行多维度综合分析",
    ]
    tool = HybridAnalysisTool()

    for idx, query in enumerate(user_queries, start=1):
        print(f"\n--- 处理第{idx}条请求 ---")
        print(f"用户问题：{query}")
        print(f"\n结果输出：\n{handle_query(query, tool)}")


if __name__ == "__main__":
    main()
