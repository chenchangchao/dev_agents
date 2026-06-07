# 【例8-5】子 Agent 并行执行与结果汇总
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch08/src/8_5_parallel_subagent_executor.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch08/src/8_5_parallel_subagent_executor.py

import threading
import time

from ch08_runtime import Tool, ask_llm, backend_name


class PolicyAgent(Tool):
    def __init__(self):
        super().__init__(name="policy_agent", description="分析最新宏观政策")

    def call(self, _: str) -> str:
        time.sleep(0.2)
        return "政策导向积极，财政刺激与减税政策持续发力。"


class SentimentAgent(Tool):
    def __init__(self):
        super().__init__(name="sentiment_agent", description="提取市场情绪")

    def call(self, _: str) -> str:
        time.sleep(0.2)
        return "市场情绪整体乐观，投资者信心显著回暖。"


class TrendAgent(Tool):
    def __init__(self):
        super().__init__(name="trend_agent", description="预测经济增长趋势")

    def call(self, _: str) -> str:
        time.sleep(0.2)
        return "预计2024年GDP增长约5.5%，主要来源于消费与出口复苏。"


class ParallelExecutor:
    def __init__(self):
        self.results: dict[str, str] = {}
        self.lock = threading.Lock()

    def run_agent(self, agent: Tool, name: str) -> None:
        print(f"→ 启动{name}")
        result = agent.call("")
        with self.lock:
            self.results[name] = result
        print(f"→ {name}完成")

    def execute_all(self, agents: dict[str, Tool]) -> dict[str, str]:
        threads = []
        for name, agent in agents.items():
            thread = threading.Thread(target=self.run_agent, args=(agent, name))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        return self.results


def build_prompt(results: dict[str, str]) -> str:
    prompt = "请根据以下信息撰写2024年一季度中国经济简报：\n"
    for title, content in results.items():
        prompt += f"\n【{title}】：{content}"
    return prompt


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动子Agent并行执行系统 ===")
    agents = {
        "政策分析": PolicyAgent(),
        "市场情绪": SentimentAgent(),
        "趋势预测": TrendAgent(),
    }
    results = ParallelExecutor().execute_all(agents)
    prompt = build_prompt(results)

    print("\n→ 汇总Prompt：")
    print(prompt)

    final_response = ask_llm(prompt, system="你是经济简报撰写助手。", max_tokens=420, label="parallel_executor")
    print("\n--- 响应输出 ---")
    print(final_response)


if __name__ == "__main__":
    main()
