# 【例6-6】Prompt 缓存与快速回放
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch06/src/6_6_prompt_cache_replay.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch06/src/6_6_prompt_cache_replay.py
#
# 缓存写入：
# ch06/data/prompt_cache.json

import hashlib

from ch06_runtime import PromptCache, Tool, ask_llm, backend_name


class EconomicIndicatorTool(Tool):
    def __init__(self):
        super().__init__(name="economic_indicator", description="宏观指标摘要工具")

    def call(self, query: str) -> str:
        return (
            f"[Tool] 针对“{query}”的指标摘要："
            "CPI约2.1%，PMI约51.4，消费同比上升约6.2%，需关注外需和地产链不确定性。"
        )


def hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()


def build_prompt(user_query: str, tool_output: str) -> str:
    system_prompt = "你是一位宏观经济分析师，回答应包含数据推理、趋势预测与政策建议。"
    return f"""【系统提示】
{system_prompt}

【外部工具输出】
{tool_output}

【用户提问】
{user_query}

请基于上述内容生成结构化分析结果。"""


def get_or_generate(prompt: str, cache: dict[str, str]) -> tuple[str, bool, str]:
    prompt_hash = hash_prompt(prompt)
    cached = cache.get(prompt_hash)
    if cached:
        return cached, True, prompt_hash

    answer = ask_llm(
        prompt,
        system="你是宏观经济分析助手，回答要结构化、谨慎。",
        max_tokens=460,
        label="prompt_cache",
    )
    cache[prompt_hash] = answer
    return answer, False, prompt_hash


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== Prompt缓存与快速回放机制测试系统 ===")

    cache_store = PromptCache("prompt_cache.json")
    cache = cache_store.load()
    tool = EconomicIndicatorTool()

    user_query = "请结合当前宏观经济环境预测2024年GDP走势，并指出主要影响因素。"
    tool_output = tool.call(user_query)
    full_prompt = build_prompt(user_query, tool_output)

    answer, hit, prompt_hash = get_or_generate(full_prompt, cache)
    cache_store.save(cache)

    print(f"→ 任务哈希ID：{prompt_hash}")
    print("\n--- 缓存命中，直接回放 ---" if hit else "\n--- 未命中缓存，调用模型生成 ---")
    print(answer)
    print(f"\n当前缓存条目数量：{len(cache)}")


if __name__ == "__main__":
    main()
