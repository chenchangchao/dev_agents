# 【例5-9】
# hybrid_text_table_rag.py
#
# 本例演示混合RAG：
# 1. 财政政策文本语义检索
# 2. 财政结构化数据查询
# 3. 将文本上下文和数据上下文一起交给LLM生成回答
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch05/src/5_9_hybrid_text_table_rag.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch05/src/5_9_hybrid_text_table_rag.py

import re

from rag_utils import SimpleVectorStore, ask_llm, backend_name


FISCAL_TEXTS = [
    "积极的财政政策要适度加力、提质增效，支持扩大内需和科技创新。",
    "财政部门将加强地方政府债务风险防控，推动债务管理更加规范透明。",
    "财政资金应更多投向民生保障、教育、医疗和重点基础设施领域。",
    "减税降费政策继续优化，重点支持小微企业和制造业发展。",
]

FISCAL_TABLE = [
    {"year": 2019, "spending_gdp": 16.8, "revenue_gdp": 14.6},
    {"year": 2020, "spending_gdp": 17.7, "revenue_gdp": 14.2},
    {"year": 2021, "spending_gdp": 16.5, "revenue_gdp": 15.1},
    {"year": 2022, "spending_gdp": 16.9, "revenue_gdp": 14.7},
]


class FiscalTextSearchTool:
    def __init__(self, docs: list[str]):
        self.store = SimpleVectorStore(docs)

    def call(self, query: str) -> str:
        results = self.store.similarity_search(query, k=3)
        return "\n".join(f"- {doc.page_content}" for doc, _ in results)


class FiscalDataQueryTool:
    def __init__(self, rows: list[dict]):
        self.rows = rows

    def call(self, query: str) -> str:
        year_match = re.search(r"(20\d{2})", query)
        year = int(year_match.group(1)) if year_match else 2020
        row = next((item for item in self.rows if item["year"] == year), None)
        if not row:
            return f"未找到{year}年的财政数据。"

        if "收入" in query:
            return f"{year}年政府收入占GDP比重为{row['revenue_gdp']}%。"
        if "支出" in query:
            return f"{year}年政府支出占GDP比重为{row['spending_gdp']}%。"
        return (
            f"{year}年政府支出占GDP比重为{row['spending_gdp']}%，"
            f"政府收入占GDP比重为{row['revenue_gdp']}%。"
        )


def build_prompt(query: str, text_context: str, data_context: str) -> str:
    return f"""你是一个财税智能助手，拥有文本政策知识和财政结构化数据。

【政策文本】
{text_context}

【财政数据】
{data_context}

【用户问题】
{query}

请结合以上信息回答，要求准确、简洁、有数据支撑。"""


if __name__ == "__main__":
    print(f"LLM后端：{backend_name()}")
    text_tool = FiscalTextSearchTool(FISCAL_TEXTS)
    data_tool = FiscalDataQueryTool(FISCAL_TABLE)

    query = "2020年中国的财政支出是多少？相关政策有没有调整说明？"
    text_context = text_tool.call(query)
    data_context = data_tool.call(query)

    print("\n用户提问：", query)
    print("\n--- 政策文本检索 ---")
    print(text_context)
    print("\n--- 结构化数据查询 ---")
    print(data_context)
    print("\n--- Agent回答 ---")
    print(
        ask_llm(
            build_prompt(query, text_context, data_context),
            system="你只能基于给定政策文本和财政数据回答，不能编造额外数字。",
            temperature=0.2,
            max_tokens=400,
        )
    )
