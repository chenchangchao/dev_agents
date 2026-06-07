# 【例5-6】
# vector_store_compare_policy_rag.py
#
# 本例演示不同向量数据库工具的统一抽象。为了本地可运行，使用SimpleVectorStore
# 模拟FAISS、Chroma、Weaviate的检索接口，而不强制安装或启动这些服务。
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch05/src/5_6_vector_store_compare_policy_rag.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch05/src/5_6_vector_store_compare_policy_rag.py

from rag_utils import SimpleVectorStore, ask_llm, backend_name


POLICY_DOCS = [
    "国家能源局提出加快推进新能源基础设施建设，提升电力系统调节能力。",
    "可再生能源消纳政策强调完善跨省区电力交易机制。",
    "新型储能发展政策鼓励独立储能参与电力市场交易。",
    "绿色低碳转型要求推动煤电灵活性改造和新能源协同发展。",
    "分布式光伏管理办法强调规范备案、电网接入和安全运行。",
]


class VectorSearchTool:
    def __init__(self, name: str, docs: list[str]):
        self.name = name
        self.store = SimpleVectorStore(docs)

    def search(self, query: str, k: int = 3) -> str:
        results = self.store.similarity_search(query, k=k)
        return "\n".join(f"- {doc.page_content}" for doc, _ in results)


def build_answer(query: str, contexts: dict[str, str]) -> str:
    merged_context = "\n\n".join(f"【{name}检索结果】\n{text}" for name, text in contexts.items())
    prompt = f"""你是一名能源政策问答助手。请基于多个向量检索工具返回的内容回答。

{merged_context}

【用户问题】
{query}

请用中文总结答案，并指出检索内容主要来自哪些政策主题。"""
    return ask_llm(
        prompt,
        system="你只能基于给定上下文回答，不要编造政策编号。",
        temperature=0.2,
        max_tokens=350,
    )


if __name__ == "__main__":
    print(f"LLM后端：{backend_name()}")
    tools = [
        VectorSearchTool("FAISS模拟", POLICY_DOCS),
        VectorSearchTool("Chroma模拟", POLICY_DOCS),
        VectorSearchTool("Weaviate模拟", POLICY_DOCS),
    ]

    query = "近期有哪些新能源相关政策？"
    contexts = {tool.name: tool.search(query) for tool in tools}

    print("\n用户问题：", query)
    for name, context in contexts.items():
        print(f"\n--- {name} ---")
        print(context)

    print("\n--- RAG回答 ---")
    print(build_answer(query, contexts))
