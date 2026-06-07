# 【例9-5】发布订阅式 Agent 广播
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch09/src/9_5_pubsub_broadcast_agents.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch09/src/9_5_pubsub_broadcast_agents.py

import time
import uuid
from collections.abc import Callable

from ch09_runtime import PromptEntry, ask_llm, backend_name


class PubSubMessage:
    def __init__(self, topic: str, content: str):
        self.id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.topic = topic
        self.content = content

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "timestamp": self.timestamp, "topic": self.topic, "content": self.content}


class SubscriberAgent:
    def __init__(self, name: str, topic_filter: Callable[[str], bool]):
        self.name = name
        self.topic_filter = topic_filter

    def on_message(self, message: PubSubMessage) -> str:
        if not self.topic_filter(message.topic):
            return f"{self.name}：忽略消息。"
        prompt = [
            PromptEntry(role="system", content=f"你是 {self.name}，负责订阅主题并分析消息。"),
            PromptEntry(role="user", content=f"请分析以下新闻：{message.content}"),
            PromptEntry(role="memory", content=f"分析主题：{message.topic}"),
        ]
        response = ask_llm(prompt, max_tokens=380, label=self.name)
        return f"{self.name} 分析结果：{response}"


class PubSubSystem:
    def __init__(self):
        self.subscribers: list[SubscriberAgent] = []

    def register(self, agent: SubscriberAgent) -> None:
        self.subscribers.append(agent)

    def publish(self, message: PubSubMessage) -> list[str]:
        print(f"\n【广播】主题：{message.topic}")
        print(f"内容：{message.content}")
        return [agent.on_message(message) for agent in self.subscribers]


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    system = PubSubSystem()
    system.register(SubscriberAgent("Finance-Agent", lambda topic: "财经" in topic))
    system.register(SubscriberAgent("AI-Finance-Agent", lambda topic: "AI" in topic or "财经" in topic))
    message = PubSubMessage("财经快讯", "中国央行宣布降准50个基点以稳定宏观经济预期，股市应声上涨。")
    results = system.publish(message)

    print("\n--- 响应结果 ---")
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
