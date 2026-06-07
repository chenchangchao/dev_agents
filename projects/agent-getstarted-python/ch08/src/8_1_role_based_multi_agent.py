# 【例8-1】基于职责建模的多 Agent 协作
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch08/src/8_1_role_based_multi_agent.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch08/src/8_1_role_based_multi_agent.py

from ch08_runtime import Tool, ask_llm, backend_name


class NewsAgent(Tool):
    def __init__(self):
        super().__init__(name="news_agent", description="抓取并提取财经新闻")

    def call(self, query: str) -> str:
        return "当前宏观新闻显示：制造业回暖，CPI保持温和，出口增长稳定。"


class ForecastAgent(Tool):
    def __init__(self):
        super().__init__(name="forecast_agent", description="根据资讯预测GDP趋势")

    def call(self, summary: str) -> str:
        return f"综合判断2024年GDP预计增长约5.3%。依据：{summary}"


class ControlAgent:
    def __init__(self, news_tool: NewsAgent, forecast_tool: ForecastAgent):
        self.news_tool = news_tool
        self.forecast_tool = forecast_tool

    def run(self, user_query: str) -> str:
        print("→ 开始执行主控Agent任务拆解与调度")
        print("→ 调用新闻Agent进行资讯提取...")
        summary = self.news_tool.call(user_query)
        print(f"→ 新闻Agent返回：{summary}")

        print("→ 调用预测Agent进行经济建模...")
        forecast = self.forecast_tool.call(summary)
        print(f"→ 预测Agent返回：{forecast}")

        final_prompt = f"""请基于以下任务链条返回统一回应：

任务目标：{user_query}
资讯摘要：{summary}
预测输出：{forecast}

请以财经顾问语气生成最终报告。"""
        return ask_llm(final_prompt, system="你是财经顾问，回答要简洁、结构化。", max_tokens=420, label="role_multi_agent")


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动基于职责建模的多Agent系统 ===")
    control_agent = ControlAgent(news_tool=NewsAgent(), forecast_tool=ForecastAgent())
    response = control_agent.run("请预测2024年中国GDP走势，并说明依据")
    print("\n--- 最终响应 ---")
    print(response)


if __name__ == "__main__":
    main()
