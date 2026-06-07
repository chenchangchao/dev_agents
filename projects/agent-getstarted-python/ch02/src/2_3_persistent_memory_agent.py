# 【例2-3】
# persistent_memory_agent.py

import os
import json
from typing import List
from agent_runtime import BaseMemory, BaseTool, LocalAgent as Agent, Tool, llm_config, memory_path


# 自定义Memory：将上下文持久化至JSON文件
class PersistentJSONMemory(BaseMemory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory_file = memory_path(f"{session_id}.json")
        self.messages = self.load()

    def load(self) -> List[dict]:
        if self.memory_file.exists():
            with self.memory_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save(self):
        with self.memory_file.open("w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def add_message(self, message: dict):
        self.messages.append(message)
        self.save()

    def get(self) -> List[dict]:
        return self.messages


# 示例工具：记录交互内容
class LogUserInputTool(BaseTool):
    def run(self, params: dict) -> str:
        content = params.get("content", "")
        return f"记录完成：{content}"

    @property
    def description(self):
        return "将用户输入写入交互日志"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "用户内容"}
            },
            "required": ["content"]
        }


# 构建Agent
def build_agent(session_id: str):
    llm = llm_config(
        temperature=0.3,
        max_tokens=512
    )

    memory = PersistentJSONMemory(session_id=session_id)

    tools = [LogUserInputTool()]

    agent = Agent(
        name="SessionRecoveryAgent",
        llm=llm,
        tools=tools,
        memory=memory,
        system_message="你是一位具备上下文持久化能力的智能体，可以记录并恢复中断的多轮交互内容"
    )

    return agent


# 模拟用户对话过程（中断→恢复）
def simulate_session():
    session_id = "user_20250501"

    agent = build_agent(session_id)

    print(agent.chat("我想订一张5月5日去上海的机票"))
    print(agent.chat("时间大约是早上8点左右"))
    print(agent.chat("帮我记录这个需求"))

    print("\n模拟中断后重新启动...\n")

    agent2 = build_agent(session_id)

    print(agent2.chat("现在继续，出发地是北京"))

if __name__ == "__main__":
    simulate_session()
