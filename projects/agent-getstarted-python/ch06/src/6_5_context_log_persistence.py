# 【例6-5】上下文日志持久化
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch06/src/6_5_context_log_persistence.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch06/src/6_5_context_log_persistence.py
#
# 日志写入：
# ch06/data/agent_context_log.jsonl

from ch06_runtime import JsonlContextLog, Tool, ask_llm, backend_name, build_segment


class EconomicPredictor(Tool):
    def __init__(self):
        super().__init__(name="economic_predict", description="用于经济数据预测的工具")

    def call(self, query: str) -> str:
        print("→ 已调用经济预测工具")
        return ask_llm(
            f"请预测并简要说明：{query}",
            system="你是经济预测工具，输出要包含趋势、依据和风险。",
            max_tokens=280,
            label="economic_predictor",
        )


def build_final_prompt(system_prompt: str, tool_output: str, user_input: str) -> str:
    return f"""系统提示：{system_prompt}
工具信息：{tool_output}
用户提问：{user_input}

请基于以上信息进行严谨回答。"""


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 持久化上下文日志系统启动 ===")

    log = JsonlContextLog("agent_context_log.jsonl")
    tool = EconomicPredictor()
    task_id = "t006"

    system_prompt = "你是中国国家财政专家助手，所有回答需简洁、严谨、有数据支撑。"
    log.append(build_segment("system", system_prompt, "system_prompt", task_id, model="system"))

    user_input = "请预测2024年中国GDP增长，并说明依据。"
    log.append(build_segment("user", user_input, "input", task_id, model="user"))

    tool_output = tool.call(user_input)
    log.append(build_segment("tool", tool_output, "tool_output", task_id, model=backend_name(), meta={"tool": tool.name}))

    final_prompt = build_final_prompt(system_prompt, tool_output, user_input)
    answer = ask_llm(
        final_prompt,
        system="你是一位宏观经济与财政政策综合分析助手。",
        max_tokens=420,
        label="context_log_answer",
    )
    log.append(build_segment("assistant", answer, "response", task_id, model=backend_name()))

    print("\n--- 最终回答输出 ---")
    print(answer)

    print("\n--- 最新上下文日志（展示最后5条） ---")
    for item in log.tail(5):
        print(item)


if __name__ == "__main__":
    main()
