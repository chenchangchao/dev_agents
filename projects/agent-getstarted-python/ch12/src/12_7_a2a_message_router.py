# 【例12-7】A2A 消息协议与模块路由
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch12/src/12_7_a2a_message_router.py

import uuid
from collections.abc import Callable
from typing import Any

from ch12_runtime import ask_llm, backend_name


class A2AMessage:
    def __init__(self, sender: str, receiver: str, payload: str, metadata: dict[str, Any] | None = None):
        self.message_id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.payload = payload
        self.metadata = metadata or {}


class Agent:
    def __init__(self, name: str):
        self.name = name
        self.handlers: dict[str, Callable[[A2AMessage], str]] = {}

    def register_handler(self, task_type: str, handler: Callable[[A2AMessage], str]) -> None:
        self.handlers[task_type] = handler

    def receive_message(self, message: A2AMessage) -> str:
        task = message.metadata.get("task_type", "default")
        handler = self.handlers.get(task)
        if not handler:
            return f"[{self.name}] 不支持的任务类型：{task}"
        return handler(message)


class ToolAgent(Agent):
    def __init__(self):
        super().__init__("ToolAgent")
        self.register_handler("tool_call", self.handle_tool)

    def handle_tool(self, msg: A2AMessage) -> str:
        tool_name = msg.metadata.get("tool", "未知工具")
        return f"[{self.name}] 已调用工具：{tool_name}，执行内容为：{msg.payload}"


class AnswerAgent(Agent):
    def __init__(self):
        super().__init__("AnswerAgent")
        self.register_handler("qa", self.handle_qa)

    def handle_qa(self, msg: A2AMessage) -> str:
        return ask_llm(f"请回答问题：{msg.payload}", temperature=0.3, max_tokens=300, label="a2a_answer")


class A2ARouter:
    def __init__(self):
        self.agents: dict[str, Agent] = {}

    def register_agent(self, agent: Agent) -> None:
        self.agents[agent.name] = agent

    def dispatch(self, msg: A2AMessage) -> str:
        if msg.receiver not in self.agents:
            return f"[ERROR] 目标Agent不存在：{msg.receiver}"
        return self.agents[msg.receiver].receive_message(msg)


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    router = A2ARouter()
    router.register_agent(ToolAgent())
    router.register_agent(AnswerAgent())
    messages = [
        A2AMessage("MainAgent", "ToolAgent", "请将CSV文件中的数据进行分析", {"task_type": "tool_call", "tool": "DataAnalyzer"}),
        A2AMessage("MainAgent", "AnswerAgent", "什么是Transformer架构？", {"task_type": "qa"}),
    ]
    for message in messages:
        print(router.dispatch(message))


if __name__ == "__main__":
    main()
