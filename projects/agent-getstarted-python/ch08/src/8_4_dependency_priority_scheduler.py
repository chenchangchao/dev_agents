# 【例8-4】带依赖和优先级的任务图调度
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch08/src/8_4_dependency_priority_scheduler.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch08/src/8_4_dependency_priority_scheduler.py

import heapq
import time
from collections.abc import Callable

from ch08_runtime import ask_llm, backend_name


class Task:
    def __init__(self, task_id: str, func: Callable[[], str], depends: list[str] | None = None, priority: int = 1):
        self.id = task_id
        self.func = func
        self.depends = depends or []
        self.priority = priority
        self.executed = False
        self.result: str | None = None


class TaskExecutor:
    def __init__(self):
        self.tasks: dict[str, Task] = {}
        self.execution_log: list[tuple[str, str]] = []

    def register(self, task: Task) -> None:
        self.tasks[task.id] = task

    def _can_execute(self, task: Task) -> bool:
        return all(self.tasks[dep].executed for dep in task.depends)

    def execute(self) -> None:
        queue = []
        for task in self.tasks.values():
            if not task.depends:
                heapq.heappush(queue, (-task.priority, task.id))

        while queue:
            _, task_id = heapq.heappop(queue)
            task = self.tasks[task_id]
            if task.executed or not self._can_execute(task):
                continue

            print(f"\n→ 执行任务：{task_id}（优先级：{task.priority}）")
            task.result = task.func()
            task.executed = True
            self.execution_log.append((task_id, task.result))

            for candidate in self.tasks.values():
                if not candidate.executed and self._can_execute(candidate):
                    heapq.heappush(queue, (-candidate.priority, candidate.id))

    def summary(self) -> str:
        return "\n".join(f"{task_id}: {result}" for task_id, result in self.execution_log)


def task_fetch_news() -> str:
    time.sleep(0.1)
    return "抓取完成：2024年CPI增速放缓、出口回升。"


def task_model_analysis() -> str:
    time.sleep(0.1)
    return "分析完成：预计GDP增长约5.4%。"


def task_generate_graph() -> str:
    time.sleep(0.1)
    return "图表渲染完成。"


def task_generate_report() -> str:
    time.sleep(0.1)
    return "最终报告生成完毕。"


def build_executor() -> TaskExecutor:
    executor = TaskExecutor()
    executor.register(Task("fetch_news", task_fetch_news, priority=3))
    executor.register(Task("model", task_model_analysis, depends=["fetch_news"], priority=4))
    executor.register(Task("graph", task_generate_graph, depends=["model"], priority=1))
    executor.register(Task("report", task_generate_report, depends=["model"], priority=2))
    return executor


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动带依赖与优先级调度的Agent任务系统 ===")
    executor = build_executor()
    executor.execute()

    prompt = f"""以下是多任务智能体执行结果，请生成一段完整分析总结：

{executor.summary()}"""
    result = ask_llm(prompt, system="你是多任务执行结果总结助手。", max_tokens=380, label="dependency_scheduler")
    print("\n--- 输出总结 ---")
    print(result)


if __name__ == "__main__":
    main()
