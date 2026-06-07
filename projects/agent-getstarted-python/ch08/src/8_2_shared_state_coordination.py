# 【例8-2】多 Agent 共享状态协作
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch08/src/8_2_shared_state_coordination.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch08/src/8_2_shared_state_coordination.py

from typing import Any

from ch08_runtime import Tool, ask_llm, backend_name


class SharedState:
    def __init__(self):
        self.state: dict[str, Any] = {}

    def update(self, key: str, value: Any) -> None:
        print(f"→ [状态更新] {key} = {value}")
        self.state[key] = value

    def get(self, key: str, default=None):
        return self.state.get(key, default)

    def dump(self) -> dict[str, Any]:
        return self.state.copy()


class AnalysisAgent(Tool):
    def __init__(self, shared_state: SharedState):
        super().__init__(name="analysis_agent", description="执行数据分析任务")
        self.shared = shared_state

    def call(self, query: str) -> str:
        result = "制造业增长、CPI温和回落、出口回升"
        self.shared.update("econ_trend", result)
        return f"[分析完成] 趋势：{result}"


class ReportAgent(Tool):
    def __init__(self, shared_state: SharedState):
        super().__init__(name="report_agent", description="根据分析内容撰写最终报告")
        self.shared = shared_state

    def call(self, _: str) -> str:
        trend = self.shared.get("econ_trend", "无数据")
        report = f"2024年经济展望：预计{trend}，将推动GDP稳定增长。"
        self.shared.update("final_report", report)
        return report


class ControllerAgent:
    def __init__(self, analysis_tool: AnalysisAgent, report_tool: ReportAgent, shared_state: SharedState):
        self.shared = shared_state
        self.analysis_tool = analysis_tool
        self.report_tool = report_tool

    def execute(self, user_query: str) -> str:
        print("\n→ 主控Agent启动，任务分析中...")
        self.shared.update("user_input", user_query)
        analysis = self.analysis_tool.call(user_query)
        print("→ 分析Agent响应：", analysis)
        report = self.report_tool.call("")
        print("→ 报告Agent响应：", report)

        prompt = f"""请基于以下共享状态生成一段完整用户报告：
用户问题：{self.shared.get('user_input')}
分析内容：{self.shared.get('econ_trend')}
最终报告：{self.shared.get('final_report')}
共享状态快照：{self.shared.dump()}"""
        return ask_llm(prompt, system="你是多Agent系统的总控报告助手。", max_tokens=420, label="shared_state")


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动多Agent状态共享系统 ===")
    shared = SharedState()
    controller = ControllerAgent(AnalysisAgent(shared), ReportAgent(shared), shared)
    final_response = controller.execute("请分析2024年中国经济走势，并撰写总结报告")
    print("\n--- 最终响应 ---")
    print(final_response)


if __name__ == "__main__":
    main()
