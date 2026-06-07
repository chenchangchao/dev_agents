# 【例8-6】并行任务上下文隔离
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch08/src/8_6_parallel_context_isolation.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch08/src/8_6_parallel_context_isolation.py

import threading

from ch08_runtime import Message, ask_llm, backend_name, format_messages


class ContextScope:
    def __init__(self):
        self.contexts: dict[str, list[Message]] = {}
        self.lock = threading.Lock()

    def create(self, task_id: str) -> None:
        with self.lock:
            self.contexts[task_id] = []

    def append(self, task_id: str, message: Message) -> None:
        with self.lock:
            if task_id in self.contexts:
                self.contexts[task_id].append(message)

    def get(self, task_id: str) -> list[Message]:
        with self.lock:
            return list(self.contexts.get(task_id, []))

    def destroy(self, task_id: str) -> None:
        with self.lock:
            self.contexts.pop(task_id, None)


class AgentTask(threading.Thread):
    def __init__(self, task_id: str, user_input: str, scope: ContextScope):
        super().__init__()
        self.task_id = task_id
        self.user_input = user_input
        self.scope = scope
        self.result = ""

    def run(self) -> None:
        print(f"→ 启动任务[{self.task_id}]")
        self.scope.create(self.task_id)
        self.scope.append(self.task_id, Message(role="user", content=self.user_input))
        history = self.scope.get(self.task_id)
        prompt = f"请基于以下独立任务上下文回答：\n{format_messages(history)}"
        self.result = ask_llm(prompt, system="你是并行任务处理助手。", max_tokens=260, label=self.task_id)
        self.scope.append(self.task_id, Message(role="assistant", content=self.result))
        print(f"→ 任务[{self.task_id}]完成")


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动多任务上下文隔离系统 ===")
    context_scope = ContextScope()
    tasks = [
        AgentTask("task_001", "请简要说明当前中国宏观经济趋势", context_scope),
        AgentTask("task_002", "总结2023年人工智能的发展方向", context_scope),
        AgentTask("task_003", "请写一段自然语言处理技术简介", context_scope),
    ]
    for task in tasks:
        task.start()
    for task in tasks:
        task.join()

    combined_prompt = "以下是三个并行任务的输出，请统一生成一段内容总结：\n"
    for task in tasks:
        combined_prompt += f"\n【{task.task_id}】：{task.result}"

    summary = ask_llm(combined_prompt, system="你是并行任务结果汇总助手。", max_tokens=420, label="context_isolation")
    print("\n--- 汇总总结 ---")
    print(summary)


if __name__ == "__main__":
    main()
