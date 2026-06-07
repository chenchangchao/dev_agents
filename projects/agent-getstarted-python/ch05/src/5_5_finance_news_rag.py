# 【例5-5】
# finance_news_rag.py
#
# 本例演示金融新闻标题检索 + LLM总结。
#
# 支持：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch05/src/5_5_finance_news_rag.py
# 或直接使用云端DeepSeek/OpenAI兼容API：
# python3 ch05/src/5_5_finance_news_rag.py

import requests

from rag_utils import SimpleVectorStore, ask_llm, backend_name


FALLBACK_NEWS = [
    "央行开展公开市场操作维护银行体系流动性合理充裕",
    "多家银行下调存款挂牌利率推动实体融资成本下降",
    "财政政策加力提效支持重点领域和薄弱环节",
    "人民币汇率保持基本稳定跨境资金流动总体平稳",
    "金融监管部门强调防范化解重点领域金融风险",
]


def fetch_finance_news() -> list[str]:
    url = "https://www.kfzx.com.cn/api/newslist?page=1&pageSize=20"
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        rows = response.json().get("data", {}).get("rows", [])
        titles = [item["Title"] for item in rows if item.get("Title")]
        return titles or FALLBACK_NEWS
    except Exception:
        return FALLBACK_NEWS


def answer_with_context(query: str, context: str) -> str:
    prompt = f"""你是一名金融新闻助手。请只基于检索到的新闻标题回答问题。

【检索结果】
{context}

【用户问题】
{query}

请用中文给出不超过150字的回答。"""
    return ask_llm(
        prompt,
        system="你是一个严谨的RAG问答助手，不要编造未出现在上下文中的事实。",
        temperature=0.2,
        max_tokens=300,
    )


if __name__ == "__main__":
    print(f"LLM后端：{backend_name()}")
    docs = fetch_finance_news()
    store = SimpleVectorStore(docs)

    query = "最近有哪些关于货币政策的消息？"
    retrieved = store.similarity_search(query, k=3)
    context = "\n".join(f"- {doc.page_content}" for doc, _ in retrieved)

    print("\n用户问题：", query)
    print("\n--- 检索结果 ---")
    print(context)
    print("\n--- RAG回答 ---")
    print(answer_with_context(query, context))
