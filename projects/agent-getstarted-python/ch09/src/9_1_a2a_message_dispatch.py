# 【例9-1】A2A 消息格式与调度
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# python3 ch09/src/9_1_a2a_message_dispatch.py

import json
import random
import uuid
from datetime import UTC, datetime
from typing import Any


def build_a2a_message(
    sender: str,
    receiver: str,
    msg_type: str,
    intent: str,
    context_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": f"msg-{datetime.now(UTC).strftime('%Y%m%d%H%M%S')}-{random.randint(100, 999)}",
        "timestamp": datetime.now(UTC).isoformat(),
        "from": sender,
        "to": receiver,
        "type": msg_type,
        "intent": intent,
        "context_id": context_id,
        "payload": payload,
        "auth": {"token": "demo-token-123456", "signature": "demo-signature-sha256"},
    }


class LocalAgent:
    def __init__(self, name: str):
        self.name = name

    def handle_message(self, message: dict[str, Any]) -> str:
        intent = message["intent"]
        task = message["payload"].get("task", "")
        params = message["payload"].get("params", {})
        print(f"\n[{self.name}] 接收到消息:")
        print(json.dumps(message, indent=2, ensure_ascii=False))

        if intent == "run_task" and task == "analyze_financial_trends":
            return f"分析完成：{params.get('region', '未知')}地区{params.get('period', '未指定')}期间经济增长放缓，通胀风险可控。"
        if intent == "query_weather":
            return f"{params.get('city', '未知')}天气晴朗，气温18-24摄氏度。"
        return "无法识别的任务指令。"


class A2AController:
    def __init__(self):
        self.agents: dict[str, LocalAgent] = {}

    def register_agent(self, agent: LocalAgent) -> None:
        self.agents[agent.name] = agent

    def dispatch(self, message: dict[str, Any]) -> str:
        target = message["to"]
        if target not in self.agents:
            return f"目标Agent {target} 不存在。"
        return self.agents[target].handle_message(message)


def main() -> None:
    print("=== A2A消息调度示例 ===")
    controller = A2AController()
    for name in ["agent:planner", "agent:executor", "agent:weather"]:
        controller.register_agent(LocalAgent(name))

    context_id = str(uuid.uuid4())
    messages = [
        build_a2a_message(
            "agent:planner",
            "agent:executor",
            "command",
            "run_task",
            context_id,
            {"task": "analyze_financial_trends", "params": {"region": "Asia", "period": "2024-Q4"}},
        ),
        build_a2a_message(
            "agent:planner",
            "agent:weather",
            "request",
            "query_weather",
            context_id,
            {"task": "query_weather", "params": {"city": "上海"}},
        ),
    ]

    for index, message in enumerate(messages, start=1):
        print(f"\n--- 调度执行消息 {index} ---")
        print(f"\n→ 执行结果：{controller.dispatch(message)}")


if __name__ == "__main__":
    main()
