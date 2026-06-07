# 【例8-3】多 Agent 任务调度策略
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch08/src/8_3_task_dispatcher_strategy.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch08/src/8_3_task_dispatcher_strategy.py

import random

from ch08_runtime import Tool, ask_llm, backend_name


class EconomicAgent(Tool):
    def __init__(self, name: str, quality: float):
        super().__init__(name=name, description=f"财经预测Agent，性能评分：{quality}")
        self.quality = quality

    def call(self, query: str) -> str:
        return f"[{self.name}] 分析完成：预计增长为 {round(5 + random.random(), 2)}%，性能评分为 {self.quality}。"


class TaskDispatcher:
    def __init__(self, agents: list[Tool], strategy: str = "round_robin", weights: dict[str, int] | None = None):
        self.agents = agents
        self.strategy = strategy
        self.weights = weights or {}
        self.counter = 0
        self.pool = self._build_weighted_pool() if strategy == "weighted" else agents

    def _build_weighted_pool(self) -> list[Tool]:
        pool = []
        for agent in self.agents:
            pool.extend([agent] * self.weights.get(agent.name, 1))
        return pool

    def dispatch(self) -> Tool:
        if self.strategy == "round_robin":
            agent = self.agents[self.counter % len(self.agents)]
            self.counter += 1
            return agent
        return random.choice(self.pool)


class ControlSystem:
    def __init__(self, dispatcher: TaskDispatcher):
        self.dispatcher = dispatcher

    def run_task(self, query: str) -> str:
        tool = self.dispatcher.dispatch()
        print(f"\n→ 当前调度策略：{self.dispatcher.strategy.upper()}")
        print(f"→ 分配给Agent：{tool.name}")
        tool_output = tool.call(query)
        prompt = f"""请根据以下Agent的分析生成最终简明财经预测：

用户问题：{query}
Agent分析：{tool_output}"""
        return ask_llm(prompt, system="你是财经预测汇总助手。", max_tokens=300, label="dispatcher")


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动任务调度系统 ===")
    random.seed(8)
    agents = [
        EconomicAgent("econ_agent_1", 0.85),
        EconomicAgent("econ_agent_2", 0.90),
        EconomicAgent("econ_agent_3", 0.95),
    ]

    round_system = ControlSystem(TaskDispatcher(agents, strategy="round_robin"))
    for _ in range(3):
        print(round_system.run_task("预测2024年中国GDP增长"))

    weighted_system = ControlSystem(
        TaskDispatcher(agents, strategy="weighted", weights={"econ_agent_1": 1, "econ_agent_2": 1, "econ_agent_3": 3})
    )
    for _ in range(3):
        print(weighted_system.run_task("预测2024年中国GDP增长"))


if __name__ == "__main__":
    main()
