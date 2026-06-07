# 【例12-2】用户意图识别与入口解析
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch12/src/12_2_intent_parser.py

import ast

from ch12_runtime import ask_llm, backend_name, extract_city

INTENT_TEMPLATE = """
请将以下用户请求进行意图识别，只返回Python字典格式，包含：
- intent：意图类别，如 查询、执行、对话、工具调用
- domain：任务领域，如 天气、数据库、文本生成、翻译
- parameters：提取的关键词参数

用户请求：
"{query}"
"""


def rule_parse(query: str) -> dict:
    if "天气" in query or "气温" in query:
        return {"intent": "查询", "domain": "天气", "parameters": {"city": extract_city(query)}}
    if "翻译" in query:
        return {"intent": "执行", "domain": "翻译", "parameters": {"text": query.split("：")[-1]}}
    if "生成" in query or "短文" in query:
        return {"intent": "执行", "domain": "文本生成", "parameters": {"topic": query}}
    if "数据库" in query or "脚本" in query:
        return {"intent": "工具调用", "domain": "数据库", "parameters": {"task": query}}
    return {"intent": "对话", "domain": "闲聊", "parameters": {}}


class ModelInvoker:
    def __init__(self, model_type: str = "primary"):
        self.model_type = model_type

    def run_intent_recognition(self, user_query: str) -> dict:
        prompt = INTENT_TEMPLATE.replace("{query}", user_query)
        try:
            response = ask_llm(prompt, temperature=0, max_tokens=300, label=self.model_type)
            return {"success": True, "data": ast.literal_eval(response.strip())}
        except Exception as exc:
            return {"success": False, "error": str(exc)}


class IntentParser:
    def __init__(self):
        self.primary = ModelInvoker("primary")
        self.backup = ModelInvoker("backup")

    def parse(self, query: str) -> dict:
        for invoker in [self.primary, self.backup]:
            result = invoker.run_intent_recognition(query)
            if result["success"] and isinstance(result["data"], dict):
                return result["data"]
        return rule_parse(query)


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    parser = IntentParser()
    queries = [
        "请帮我查一下今天上海的天气",
        "生成一篇关于人工智能未来发展的短文",
        "将这段文字翻译成英文：你好世界",
        "帮我执行一下数据库清理脚本",
        "你好，你能陪我聊聊天吗？",
    ]
    for index, query in enumerate(queries, start=1):
        print(f"\n用户请求{index}：{query}")
        print("识别结果：", parser.parse(query))


if __name__ == "__main__":
    main()
