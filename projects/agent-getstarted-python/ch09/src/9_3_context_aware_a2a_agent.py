# 【例9-3】上下文感知 A2A Agent
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch09/src/9_3_context_aware_a2a_agent.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch09/src/9_3_context_aware_a2a_agent.py

import time
import uuid
from typing import Any

from ch09_runtime import PromptEntry, ask_llm, backend_name


class A2AMessage:
    def __init__(self, sender: str, receiver: str, intent: str, payload: dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.sender = sender
        self.receiver = receiver
        self.intent = intent
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "from": self.sender,
            "to": self.receiver,
            "intent": self.intent,
            "payload": self.payload,
        }


class ContextAwareAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def handle_message(self, message: A2AMessage) -> str:
        task = message.payload.get("task", "")
        content = message.payload.get("content", "")
        prompt = [
            PromptEntry(role="system", content=f"你是 {self.agent_name}，负责理解上下文并完成任务。"),
            PromptEntry(role="user", content=f"请分析以下任务：{task}"),
            PromptEntry(role="memory", content=f"历史背景资料：{content}"),
        ]
        return ask_llm(prompt, max_tokens=420, label=self.agent_name)


class AgentOrchestrator:
    def __init__(self):
        self.agents: dict[str, ContextAwareAgent] = {}

    def register_agent(self, agent_name: str, agent_obj: ContextAwareAgent) -> None:
        self.agents[agent_name] = agent_obj

    def dispatch_task(self, message: A2AMessage) -> str:
        if message.receiver not in self.agents:
            return f"目标Agent [{message.receiver}] 不存在。"
        return self.agents[message.receiver].handle_message(message)


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    orchestrator = AgentOrchestrator()
    orchestrator.register_agent("agent:finance", ContextAwareAgent("agent:finance"))
    orchestrator.register_agent("agent:tech", ContextAwareAgent("agent:tech"))

    tasks = [
        A2AMessage(
            "agent:planner",
            "agent:finance",
            "run_task",
            {"task": "分析2024年中国宏观经济走势", "content": "受益于内需恢复与出口回升，GDP预计实现5%以上增长。"},
        ),
        A2AMessage(
            "agent:planner",
            "agent:tech",
            "run_task",
            {"task": "预测2025年人工智能行业发展趋势", "content": "多模态模型与具身智能的结合将成为发展重点。"},
        ),
    ]

    for index, task in enumerate(tasks, start=1):
        print(f"\n--- 执行任务 {index} ---")
        print("→ 结果：", orchestrator.dispatch_task(task))


if __name__ == "__main__":
    main()
