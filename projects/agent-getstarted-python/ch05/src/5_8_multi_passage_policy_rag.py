# 【例5-8】
# multi_passage_policy_rag.py
#
# 本例演示Top-K多段检索融合：检索多个片段后合并为上下文，再交给LLM回答。
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch05/src/5_8_multi_passage_policy_rag.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch05/src/5_8_multi_passage_policy_rag.py

from rag_utils import SimpleVectorStore, ask_llm, backend_name


POLICY_DOCS = [
    "国内生产总值核算方法说明强调生产法、收入法和支出法之间的关系。",
    "地区生产总值统一核算改革要求提升地区GDP数据的可比性和一致性。",
    "统计执法监督规定强调防范统计造假、弄虚作假。",
    "国民经济行业分类标准用于规范企业和产业统计口径。",
    "数据质量评估制度要求加强源头数据审核和统计调查全过程质量控制。",
    "季度GDP核算解释说明介绍初步核算、最终核实和数据修订流程。",
]


def split_chunks(texts: list[str], max_len: int = 100) -> list[str]:
    chunks = []
    for text in texts:
        if len(text) <= max_len:
            chunks.append(text)
        else:
            chunks.extend(text[i : i + max_len] for i in range(0, len(text), max_len))
    return chunks


def build_prompt(query: str, context: str) -> str:
    return f"""你是一名统计政策问答助手，以下是从政策数据库中检索到的多段内容：

【语料片段】
{context}

【提问】
{query}

请基于上述内容准确回答，用中文简洁清晰地作答。"""


if __name__ == "__main__":
    print(f"LLM后端：{backend_name()}")
    chunks = split_chunks(POLICY_DOCS)
    store = SimpleVectorStore(chunks)

    query = "国家统计局有哪些针对GDP核算方法的文件或解释？"
    retrieved = store.similarity_search(query, k=5)
    fused_context = "\n".join(f"- {doc.page_content}" for doc, _ in retrieved)

    print("\n用户提问：", query)
    print("\n--- Top-K融合上下文 ---")
    print(fused_context)
    print("\n--- Agent回答 ---")
    print(
        ask_llm(
            build_prompt(query, fused_context),
            system="你只能基于给定上下文回答，不能编造文件名。",
            temperature=0.2,
            max_tokens=350,
        )
    )
