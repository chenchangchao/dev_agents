# 【例9-2】Agent 注册中心、认证与能力校验
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# python3 ch09/src/9_2_agent_registry_auth.py

import hashlib
import time
import uuid
from typing import Any


class AgentRegistry:
    def __init__(self):
        self.agents: dict[str, dict[str, Any]] = {}

    def register_agent(self, name: str, token: str, signature: str, capabilities: list[str]) -> None:
        self.agents[name] = {"token": token, "signature": signature, "capabilities": capabilities}
        print(f"注册Agent：{name}，能力：{capabilities}")

    def is_authenticated(self, name: str, token: str, signature: str) -> bool:
        agent = self.agents.get(name)
        return bool(agent and agent["token"] == token and agent["signature"] == signature)

    def has_capability(self, name: str, task: str) -> bool:
        agent = self.agents.get(name)
        return bool(agent and task in agent["capabilities"])


def gen_token(agent_name: str) -> str:
    return hashlib.sha256(agent_name.encode()).hexdigest()


def build_secure_message(
    sender: str,
    receiver: str,
    task: str,
    params: dict[str, Any],
    token: str,
    signature: str,
) -> dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "from": sender,
        "to": receiver,
        "intent": "run_task",
        "payload": {"task": task, "params": params},
        "auth": {"token": token, "signature": signature},
    }


class LocalAgent:
    def __init__(self, name: str):
        self.name = name

    def handle(self, message: dict[str, Any]) -> str:
        task = message["payload"]["task"]
        if task == "text_summary":
            return "执行任务：文本摘要完成。"
        if task == "translate_text":
            return "执行任务：翻译已完成。"
        return "未知任务。"


def dispatch(registry: AgentRegistry, agent_name: str, message: dict[str, Any]) -> str:
    if not registry.is_authenticated(agent_name, message["auth"]["token"], message["auth"]["signature"]):
        return f"Agent {agent_name} 认证失败，拒绝执行。"
    if not registry.has_capability(agent_name, message["payload"]["task"]):
        return f"Agent {agent_name} 不具备执行 [{message['payload']['task']}] 的能力。"
    return LocalAgent(agent_name).handle(message)


def main() -> None:
    print("=== Agent注册与认证示例 ===")
    registry = AgentRegistry()
    registry.register_agent("agent:qwen3", gen_token("agent:qwen3"), "sig-qwen", ["text_summary", "translate_text"])
    registry.register_agent("agent:deepseek", gen_token("agent:deepseek"), "sig-deep", ["translate_text"])

    msg1 = build_secure_message(
        "agent:planner",
        "agent:qwen3",
        "text_summary",
        {"text": "本项目聚焦于多智能体系统的通信机制"},
        gen_token("agent:qwen3"),
        "sig-qwen",
    )
    msg2 = build_secure_message(
        "agent:planner",
        "agent:deepseek",
        "text_summary",
        {"text": "请将此句生成摘要"},
        gen_token("agent:deepseek"),
        "sig-deep",
    )

    print("\n--- 调度消息 1 ---")
    print("→ 执行结果：", dispatch(registry, "agent:qwen3", msg1))
    print("\n--- 调度消息 2 ---")
    print("→ 执行结果：", dispatch(registry, "agent:deepseek", msg2))


if __name__ == "__main__":
    main()
