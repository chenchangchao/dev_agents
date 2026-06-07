# 【例7-1】具备状态管理的 Agent 生命周期
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch07/src/7_1_stateful_agent_lifecycle.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch07/src/7_1_stateful_agent_lifecycle.py

import time
import uuid
from enum import Enum

from ch07_runtime import Tool, ask_llm, backend_name


class AgentState(Enum):
    INITIALIZED = "initialized"
    READY = "ready"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    COMPLETED = "completed"
    FAILED = "failed"


class ForecastTool(Tool):
    def __init__(self):
        super().__init__(name="economic_forecast", description="经济预测工具")

    def call(self, query: str) -> str:
        print("→ 调用经济预测工具...")
        return (
            f"[Tool] 针对“{query}”的预测：2024年GDP增速中枢约为5%左右，"
            "主要受消费修复、制造业投资、外需波动和政策支持影响。"
        )


class StatefulAgent:
    def __init__(self, tool: Tool):
        self.task_id = str(uuid.uuid4())
        self.state = AgentState.INITIALIZED
        self.history: list[dict] = []
        self.tool = tool

    def update_state(self, new_state: AgentState) -> None:
        print(f"→ 状态变更：{self.state.value} → {new_state.value}")
        self.state = new_state
        self.history.append({"timestamp": time.time(), "task_id": self.task_id, "state": self.state.value})

    def execute(self, system_prompt: str, user_question: str) -> str:
        try:
            self.update_state(AgentState.READY)
            self.update_state(AgentState.RUNNING)
            self.update_state(AgentState.WAITING_TOOL)
            tool_result = self.tool.call(user_question)
            self.update_state(AgentState.RUNNING)

            merged_prompt = f"""【系统提示】
{system_prompt}

【用户提问】
{user_question}

【工具响应】
{tool_result}

请生成结构化回答，包含趋势、依据和风险。"""
            answer = ask_llm(
                merged_prompt,
                system="你是一名宏观经济顾问，回答需严谨、简明。",
                max_tokens=460,
                label="stateful_agent",
            )
            self.update_state(AgentState.COMPLETED)
            return answer
        except Exception as exc:
            self.update_state(AgentState.FAILED)
            return f"[ERROR] 任务执行失败：{exc}"

    def show_history(self) -> None:
        print("\n--- 状态变更日志 ---")
        for entry in self.history:
            readable = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(entry["timestamp"]))
            print(f"{readable} | 状态：{entry['state']}")


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动具备状态管理的智能体系统 ===")
    agent = StatefulAgent(tool=ForecastTool())
    output = agent.execute(
        "你是一名宏观经济顾问，所有回答需结合当前CPI和GDP数据，表达严谨、简明。",
        "请预测2024年GDP增长趋势，并说明主要影响因素。",
    )
    print("\n--- 模型响应输出 ---")
    print(output)
    agent.show_history()


if __name__ == "__main__":
    main()
