# 【例6-3】多任务拆分与模块路由
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch06/src/6_3_multi_task_router.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch06/src/6_3_multi_task_router.py

from ch06_runtime import Tool, ask_llm, backend_name


def determine_route(user_input: str) -> str:
    if "政策" in user_input or "法律" in user_input:
        return "policy_model"
    if "数据" in user_input or "预测" in user_input or "GDP" in user_input:
        return "data_model"
    if "分析" in user_input or "工具" in user_input:
        return "tool"
    return "default"


class DataSummaryTool(Tool):
    def __init__(self):
        super().__init__(name="data_summary", description="结构化数据总结工具")

    def call(self, query: str) -> str:
        return (
            "[Tool] 已根据结构化财政数据完成分析："
            "2023年财政支出同比增长约6.8%，民生与基础设施相关支出占比较高。"
        )


def call_policy_model(prompt: str) -> str:
    return ask_llm(
        f"请作为政策专家简要说明：{prompt}",
        system="你负责政策解释，回答要聚焦政策方向和社会影响。",
        max_tokens=300,
        label="policy_model",
    )


def call_data_model(prompt: str) -> str:
    return ask_llm(
        f"请作为数据分析助手回答：{prompt}",
        system="你负责经济数据预测，回答要说明假设和不确定性。",
        max_tokens=300,
        label="data_model",
    )


def execute_task(task: str, tool: DataSummaryTool) -> str:
    route = determine_route(task)
    print(f"→ 任务路由：{route} | {task}")

    if route == "policy_model":
        return call_policy_model(task)
    if route == "data_model":
        return call_data_model(task)
    if route == "tool":
        return tool.call(task)
    return ask_llm(task, system="你是默认任务处理助手。", max_tokens=260, label="default")


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    tool = DataSummaryTool()
    sub_tasks = [
        "分析2023年的财政数据",
        "预测2024年中国GDP的增长情况",
        "说明近期的主要财政政策",
    ]

    responses = [execute_task(task, tool) for task in sub_tasks]

    print("\n--- 多模块合并响应 ---")
    for idx, response in enumerate(responses, start=1):
        print(f"[子任务{idx}] {response}")


if __name__ == "__main__":
    main()
