# 【例7-4】带窗口记忆机制的 Agent
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch07/src/7_4_window_memory_agent.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch07/src/7_4_window_memory_agent.py

from ch07_runtime import ConversationWindowMemory, Tool, ask_llm, backend_name


class FakeDataFetcher(Tool):
    def __init__(self):
        super().__init__(name="fake_data_fetcher", description="宏观经济数据查询工具")

    def call(self, query: str) -> str:
        if "2023" in query:
            return "2023年GDP增长率为5.2%，主要受益于消费回暖和服务业恢复。"
        if "2022" in query:
            return "2022年GDP增长为3.0%，疫情影响下内需偏弱。"
        return "暂无直接数据，请结合历史趋势审慎判断。"


class MemoryAgent:
    def __init__(self, memory: ConversationWindowMemory):
        self.memory = memory
        self.fetcher = FakeDataFetcher()

    def ask(self, user_input: str) -> str:
        tool_result = self.fetcher.call(user_input)
        prompt = f"""以下是最近对话历史：
{self.memory.text()}

当前工具查询结果：
{tool_result}

用户现在的问题是：
{user_input}

请结合历史、工具结果与常识进行回答。"""
        answer = ask_llm(prompt, system="你是带窗口记忆的宏观经济助手。", max_tokens=400, label="window_memory")
        self.memory.add("用户", user_input)
        self.memory.add("助手", answer)
        return answer


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动带窗口记忆机制的智能体 ===")
    bot = MemoryAgent(memory=ConversationWindowMemory(k=3))

    for index, question in enumerate(
        ["2022年GDP是多少？", "那2023年呢？", "你觉得2024年增长是否会更快？", "依据是什么？"],
        start=1,
    ):
        print(f"\n[Round {index}] 用户提问：{question}")
        print(bot.ask(question))


if __name__ == "__main__":
    main()
