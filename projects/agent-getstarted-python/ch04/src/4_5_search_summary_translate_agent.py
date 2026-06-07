# 【例4-5】
# 4_5_search_summary_translate_agent.py
#
# 本例演示工具链式组合：
# 1. 模拟Web搜索
# 2. 对搜索结果做一句话摘要
# 3. 如有需要再翻译成中文
#
# 本地Ollama运行：
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch04/src/4_5_search_summary_translate_agent.py
#
# 云端DeepSeek/OpenAI兼容API运行：
# python3 ch04/src/4_5_search_summary_translate_agent.py

from llm_backend import ask_text, backend_name


# 工具1：网络搜索接口（模拟）
def search_web(query: str) -> str:
    """根据用户输入执行Web搜索，并返回摘要结果"""
    if "AI Agent" in query:
        return "AI Agent是一种基于大语言模型的自主任务执行单元，具备工具调用与上下文感知能力。"
    return f"未找到关于：{query} 的相关信息"


# 工具2：摘要生成器
def summarize_text(text: str) -> str:
    """对给定文本执行摘要处理"""
    return ask_text(
        f"请用一句话总结下面内容：\n{text}",
        system="你是一个中文摘要助手。只输出一句中文摘要，不要输出思考过程。",
        temperature=0,
        max_tokens=120,
    )


# 工具3：翻译模块
def translate_to_zh(text: str) -> str:
    """将英文摘要翻译成中文"""
    return ask_text(
        f"请将下面内容翻译成中文，保持简洁：\n{text}",
        system="你是一个翻译助手。只输出中文译文，不要输出思考过程。",
        temperature=0,
        max_tokens=160,
    )


def run_agent(user_input: str) -> str:
    search_result = search_web(user_input)
    summary = summarize_text(search_result)
    if any("\u4e00" <= char <= "\u9fff" for char in summary):
        return summary
    return translate_to_zh(summary)


def main():
    print(f"LLM后端：{backend_name()}\n")
    print("用户输入：用中文总结AI Agent的基本定义")
    output = run_agent("用中文总结AI Agent的基本定义")
    print("模型输出：", output)


if __name__ == "__main__":
    main()
