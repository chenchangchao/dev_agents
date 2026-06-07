# 【例5-7】
# legal_rag_retrieval_qa.py
#
# 本例演示法律条文RAG：本地条文 -> 切片 -> 检索 -> LLM基于条文回答。
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch05/src/5_7_legal_rag_retrieval_qa.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch05/src/5_7_legal_rag_retrieval_qa.py

from rag_utils import SimpleVectorStore, ask_llm, backend_name, split_text_by_sentences


LAW_SECTIONS = [
    "企业事业单位和其他生产经营者应当防止、减少环境污染和生态破坏，对所造成的损害依法承担责任。",
    "排放污染物的企业事业单位，应当建立环境保护责任制度，明确单位负责人和相关人员的责任。",
    "重点排污单位应当按照国家有关规定和监测规范安装使用监测设备，保证监测设备正常运行。",
    "企业事业单位超过污染物排放标准或者超过重点污染物排放总量控制指标排放污染物的，应当依法限期治理。",
    "对造成环境污染事故的单位，依法追究法律责任；构成犯罪的，依法追究刑事责任。",
]


def build_chunks(sections: list[str]) -> list[str]:
    chunks = []
    for section in sections:
        chunks.extend(split_text_by_sentences(section, max_len=120, overlap=0))
    return chunks


def build_prompt(query: str, context: str) -> str:
    return f"""你是一名法律智能助手。请根据以下法律条文内容回答用户问题。

【法律条文】
{context}

【用户提问】
{query}

请结合条文内容给出明确、简洁、准确的回答。"""


if __name__ == "__main__":
    print(f"LLM后端：{backend_name()}")
    chunks = build_chunks(LAW_SECTIONS)
    store = SimpleVectorStore(chunks)

    question = "环境保护法中对企业排污行为有什么规定？"
    retrieved = store.similarity_search(question, k=4)
    context = "\n".join(f"- {doc.page_content}" for doc, _ in retrieved)

    print("\n用户提问：", question)
    print("\n--- 检索条文 ---")
    print(context)
    print("\n--- Agent回答 ---")
    print(
        ask_llm(
            build_prompt(question, context),
            system="你只能根据提供的法律条文回答，不能编造法条编号。",
            temperature=0.2,
            max_tokens=350,
        )
    )
