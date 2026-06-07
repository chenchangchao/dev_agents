# 【例7-3】带异常处理和降级流程的 Agent
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch07/src/7_3_robust_agent_error_fallback.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch07/src/7_3_robust_agent_error_fallback.py

import random

from ch07_runtime import Tool, ask_llm, backend_name


class RiskyEconomicTool(Tool):
    def __init__(self, fail_rate: float = 0.5):
        super().__init__(name="unstable_forecast", description="可能失败的经济预测工具")
        self.fail_rate = fail_rate

    def call(self, inputs: dict) -> str:
        region = inputs.get("region")
        if not region:
            raise ValueError("缺少region字段")
        if random.random() < self.fail_rate:
            raise ConnectionError("远程API调用失败：连接超时")
        return f"[Tool] 预测：{region} 2024年GDP增长预计为5.3%左右。"


class RobustAgent:
    def __init__(self, tool: Tool):
        self.tool = tool

    def process(self, user_input: str) -> str:
        region = "美国" if "美国" in user_input else "中国"
        try:
            tool_response = self.tool.call({"region": region})
            print(f"\n→ 工具返回结果：{tool_response}")
            prompt = f"""根据以下数据生成经济分析总结：
工具输出：{tool_response}
分析要求：语言简洁、观点明确、适合决策参考。"""
        except Exception as exc:
            print(f"\n→ 工具调用失败：{exc}")
            print("→ 进入降级流程，切换为模型直接生成")
            prompt = f"外部工具调用失败。请直接基于常识与历史数据，预测{region}在2024年的经济走势，并给出简明解释。"

        return ask_llm(prompt, system="你是稳健的经济分析助手。", max_tokens=380, label="robust_agent")


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动带异常处理的Agent系统 ===")
    random.seed(7)
    agent = RobustAgent(tool=RiskyEconomicTool())
    response = agent.process("请预测2024年美国的GDP增长趋势")
    print("\n--- 最终输出 ---")
    print(response)


if __name__ == "__main__":
    main()
