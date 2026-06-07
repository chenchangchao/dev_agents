# 【例2-4】
# agent_shutdown_cleanup.py

import json
import datetime
from typing import List
from agent_runtime import BaseMemory, BaseTool, LocalAgent as Agent, llm_config, cleanup_path

# 自定义Memory类，支持写入日志文件
class LoggingMemory(BaseMemory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory_file = cleanup_path(f"{session_id}.json")
        self.logs = []

    def add_message(self, message: dict):
        self.logs.append(message)

    def get(self) -> List[dict]:
        return self.logs

    def save(self):
        with self.memory_file.open("w", encoding="utf-8") as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

    def clear(self):
        if self.memory_file.exists():
            self.memory_file.unlink()

# 工具：模拟任务处理
class TaskTool(BaseTool):
    def run(self, params: dict) -> str:
        task = params.get("task", "")
        return f"任务『{task}』已完成"

    @property
    def description(self):
        return "处理一个示例任务"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "任务内容"}
            },
            "required": ["task"]
        }

# Agent对象构建
class ManagedAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = LoggingMemory(session_id)
        self.agent = Agent(
            name="CleanableAgent",
            llm=llm_config(
                temperature=0.2,
                max_tokens=256
            ),
            tools=[TaskTool()],
            memory=self.memory,
            system_message="你是一位临时任务助手，任务完成后需要注销并清理所有资源"
        )

    def chat(self, user_input: str) -> str:
        result = self.agent.chat(user_input)
        return result

    def shutdown(self):
        print(">> 执行Agent注销流程...")
        self.memory.save()
        self.memory.clear()
        log_file = cleanup_path(f"{self.session_id}_log.txt")
        with log_file.open("w", encoding="utf-8") as f:
            for item in self.memory.logs:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] {item}\n")
        print(">> Agent已成功注销并释放资源")

# 模拟流程：运行→完成→注销
def run_task_session():
    session_id = "temp_agent_001"
    agent = ManagedAgent(session_id)

    print(agent.chat("请帮我完成今天的日报任务"))
    print(agent.chat("现在关闭智能体"))

    agent.shutdown()

if __name__ == "__main__":
    run_task_session()
