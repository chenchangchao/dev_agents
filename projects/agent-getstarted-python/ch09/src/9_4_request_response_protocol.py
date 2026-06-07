# 【例9-4】A2A 请求-响应协议
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch09/src/9_4_request_response_protocol.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch09/src/9_4_request_response_protocol.py

import time
import uuid
from typing import Any

from ch09_runtime import PromptEntry, ask_llm, backend_name


class A2AMessage:
    def __init__(self, sender: str, receiver: str, msg_type: str, payload: dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.sender = sender
        self.receiver = receiver
        self.type = msg_type
        self.payload = payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "from": self.sender,
            "to": self.receiver,
            "type": self.type,
            "payload": self.payload,
        }


class RequestAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def build_request(self, task: str, question: str) -> A2AMessage:
        return A2AMessage(self.agent_name, "agent:responder", "request", {"task": task, "question": question})

    def handle_response(self, response: A2AMessage) -> None:
        print(f"\n【{self.agent_name}】收到响应：")
        print(f"任务：{response.payload['task']}")
        print(f"回答：{response.payload['answer']}")


class ResponderAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name

    def handle_request(self, message: A2AMessage) -> A2AMessage:
        task = message.payload["task"]
        question = message.payload["question"]
        prompt = [
            PromptEntry(role="system", content="你是一名专业问答机器人，回答简洁准确。"),
            PromptEntry(role="user", content=question),
            PromptEntry(role="memory", content=f"任务类别：{task}"),
        ]
        answer = ask_llm(prompt, max_tokens=360, label=self.agent_name)
        return A2AMessage(self.agent_name, message.sender, "response", {"task": task, "answer": answer})


class Dispatcher:
    def __init__(self):
        self.request_agent = RequestAgent("agent:requester")
        self.responder_agent = ResponderAgent("agent:responder")

    def run(self) -> None:
        question = "当前欧洲的通货膨胀情况如何？"
        task = "财经问答"
        request_msg = self.request_agent.build_request(task, question)
        print(f"\n【{request_msg.sender}】发出请求：{question}")
        response_msg = self.responder_agent.handle_request(request_msg)
        self.request_agent.handle_response(response_msg)


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    Dispatcher().run()


if __name__ == "__main__":
    main()
