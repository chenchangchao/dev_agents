# 【例5-3】
# rag_retrieval_demo.py
#
# 本例演示一个最小RAG召回流程：构造知识库、建立向量索引、Top-K召回。
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# python3 ch05/src/5_3_rag_retrieval_demo.py

from rag_utils import Document, SimpleVectorStore, print_search_results


documents = [
    Document(page_content="LangChain是用于构建语言模型应用的开发框架"),
    Document(page_content="向量数据库支持语义搜索与高效索引管理"),
    Document(page_content="RAG机制结合检索系统与生成模型进行问答"),
    Document(page_content="OpenAI模型支持函数调用与多轮对话"),
    Document(page_content="嵌入模型用于将文本转为向量表达形式"),
]

db = SimpleVectorStore(documents)
query = "什么是RAG机制？"
matched_docs = db.similarity_search(query, k=2)

print("查询：", query)
print_search_results(matched_docs)
