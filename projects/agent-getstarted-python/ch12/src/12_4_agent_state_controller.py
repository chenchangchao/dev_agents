# 【例12-4】Agent 子系统状态管理与调度
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch12/src/12_4_agent_state_controller.py

import uuid

from ch12_runtime import ask_llm, backend_name


class AgentState:
    def __init__(self, name: str):
        self.name = name
        self.status = "idle"
        self.last_result = None
        self.last_error = None

    def update(self, status: str, result=None, error=None) -> None:
        self.status = status
        self.last_result = result
        self.last_error = error


class AgentController:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.agents = {
            "intent": AgentState("意图识别Agent"),
            "qa": AgentState("问答Agent"),
            "tool": AgentState("工具调用Agent"),
            "dialog": AgentState("对话Agent"),
        }

    def _execute_with_fallback(self, prompt: str) -> str:
        return ask_llm(prompt, temperature=0.4, max_tokens=400, label="state_controller")

    def dispatch_task(self, task_type: str, input_text: str) -> str:
        agent = self.agents.get(task_type)
        if not agent:
            return f"未找到任务类型：{task_type}"
        agent.update("running")
        try:
            prompts = {
                "intent": f"请识别该用户输入的意图：{input_text}",
                "qa": f"请回答该问题：{input_text}",
                "tool": f"是否需要调用工具来完成该任务？{input_text}",
                "dialog": f"请以自然语言与用户继续交谈：{input_text}",
            }
            result = self._execute_with_fallback(prompts[task_type])
            agent.update("success", result=result)
            return result
        except Exception as exc:
            agent.update("failed", error=str(exc))
            return f"[ERROR] {agent.name}执行失败：{exc}"

    def get_status_snapshot(self) -> dict[str, str]:
        return {key: value.status for key, value in self.agents.items()}


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    controller = AgentController()
    tasks = [
        ("intent", "我想查一下北京的天气"),
        ("tool", "现在外面温度是多少？"),
        ("qa", "Transformer的注意力机制原理是什么？"),
        ("dialog", "你好，可以和我聊聊人工智能吗？"),
    ]
    for index, (task_type, text) in enumerate(tasks, start=1):
        print(f"\n任务{index}：{task_type} -> {text}")
        print("任务输出：", controller.dispatch_task(task_type, text))
    print("\n【Agent状态快照】")
    print(controller.get_status_snapshot())


if __name__ == "__main__":
    main()
