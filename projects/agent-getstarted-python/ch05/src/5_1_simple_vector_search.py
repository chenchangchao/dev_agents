# 【例5-1】
# simple_vector_search.py
#
# 本例演示不依赖FAISS和外部Embedding API的最小向量检索流程。
# 本地Ollama/云端API不参与本例；这里只关注“文本 -> 向量 -> 相似度检索”。
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# python3 ch05/src/5_1_simple_vector_search.py

from rag_utils import Document, SimpleVectorStore, print_search_results


sample_texts = [
    "人工智能是研究人类智能的模拟技术",
    "大语言模型具备自然语言生成能力",
    "向量数据库实现高效相似性匹配",
    "嵌入模型用于将文本转化为向量",
    "LangChain用于构建语言模型应用",
]

documents = [Document(page_content=text) for text in sample_texts]
vector_store = SimpleVectorStore(documents)

query = "什么是向量数据库？"
results = vector_store.similarity_search(query, k=3)

print("查询：", query)
print_search_results(results)
