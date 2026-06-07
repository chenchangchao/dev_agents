# 第5章 RAG机制：检索增强智能体
# 【例5-1】
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# 构造文本列表
sample_texts = [
    "人工智能是研究人类智能的模拟技术",
    "大语言模型具备自然语言生成能力",
    "向量数据库实现高效相似性匹配",
    "嵌入模型用于将文本转化为向量",
    "LangChain用于构建语言模型应用"
]

# 构造文档对象
docs = [Document(page_content=text) for text in sample_texts]

# 初始化嵌入模型
embedding_model = OpenAIEmbeddings()

# 构建FAISS向量数据库
vector_store = FAISS.from_documents(docs, embedding_model)

# 查询文本
query = "什么是向量数据库？"
results = vector_store.similarity_search(query)

# 输出结果
for i, r in enumerate(results):
    print(f"[匹配结果{i+1}]：{r.page_content}")




# 【例5-2】
# 导入必要模块
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 原始中文文档文本
document_text = """
人工智能是计算机科学的一个分支，旨在研究和开发模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。
它是一门综合性很强的学科，涉及哲学、数学、经济学、神经科学、计算机科学等多个领域。
随着计算能力和数据规模的迅猛增长，人工智能技术在自然语言处理、计算机视觉、智能控制等方面取得了显著进展。
其中，大语言模型的兴起极大提升了机器处理自然语言的能力。
然而，为了使这些模型在问答、对话等场景下具备更强的事实性与可控性，往往需要引入检索增强机制（RAG）以结合外部知识源。
在此过程中，文档切片（chunking）策略成为RAG系统性能优化的重要一环。
"""

# 实例化切片器，设置最大块长100字符，块间重叠20字符
splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

# 执行切片
chunks = splitter.split_text(document_text)

# 输出所有切片结果
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")




# 【例5-3】
# 导入必要库
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema import Document

# 加载嵌入模型
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# 准备模拟知识库
documents = [
    Document(page_content="LangChain是用于构建语言模型应用的开发框架"),
    Document(page_content="向量数据库支持语义搜索与高效索引管理"),
    Document(page_content="RAG机制结合检索系统与生成模型进行问答"),
    Document(page_content="OpenAI模型支持函数调用与多轮对话"),
    Document(page_content="嵌入模型用于将文本转为向量表达形式"),
]

# 构建向量数据库
db = FAISS.from_documents(documents, embedding_model)
# 用户查询
query = "什么是RAG机制？"
# 执行语义相似度召回
matched_docs = db.similarity_search(query, k=2)
# 输出结果
for idx, doc in enumerate(matched_docs):
    print(f"[Top {idx+1}] {doc.page_content}")




# 【例5-4】
import re
import pandas as pd

# 模拟原始文本
raw_text = """
近年来，人工智能在各个领域取得了飞速发展，尤其是在自然语言处理方面。
大语言模型（如GPT、BERT等）的出现，使得计算机在理解和生成自然语言的能力上有了质的飞跃。
然而，模型的性能高度依赖于输入数据的质量，如何对原始文本进行高质量的清洗与切分，成为构建高效语义检索系统的基础。
以RAG为代表的检索增强生成方法，需要将文档划分为结构合理、语义完整的片段，才能在后续嵌入计算与召回阶段获得最优表现。
本文将探讨一种面向中文文本的清洗与句元切分方案。
"""

# 文本清洗函数
def clean_text(text):
    text = re.sub(r"[^\u4e00-\u9fa5。！？；，、（）——“”《》：A-Za-z0-9]", "", text)
    text = re.sub(r"\s+", "", text)
    return text

# 切分函数：按中文标点断句
def split_sentences(text):
    return re.split(r"(?<=[。！？])", text)

# Chunk生成：按照最大字符数拼接
def generate_chunks(sentences, max_len=60):
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= max_len:
            current_chunk += sentence
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

# 执行清洗与切分
cleaned_text = clean_text(raw_text)
sentences = split_sentences(cleaned_text)
chunks = generate_chunks(sentences, max_len=80)

# 构造结果DataFrame
df_chunks = pd.DataFrame({"chunk_id": range(1, len(chunks) + 1), "text_chunk": chunks})

# 展示结果
import ace_tools as tools; tools.display_dataframe_to_user(name="文本清洗与句元切分结果", dataframe=df_chunks)




# 【例5-5】
import os
import faiss
import numpy as np
from typing import List
from qwen_embedding_client import QwenEmbeddingClient  # 假设为官方接口封装类
from qwen_agent.agent import Agent  # Qwen Agent框架
from qwen_agent.tools import Tool
from qwen_agent.context import Message
import requests
import json

# 设定Embedding模型API地址及密钥
QWEN_EMBEDDING_API = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
API_KEY = os.getenv("DASHSCOPE_API_KEY")

# 加载真实金融语料数据集（示例使用国开证券官网JSON接口）
def fetch_finance_news() -> List[str]:
    url = "https://www.kfzx.com.cn/api/newslist?page=1&pageSize=20"
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    news_items = resp.json().get("data", {}).get("rows", [])
    return [item["Title"] for item in news_items if "Title" in item]

# Qwen嵌入编码函数
def get_embeddings(texts: List[str]) -> List[np.ndarray]:
    payload = {
        "model": "text-embedding-v1",
        "input": texts
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    response = requests.post(QWEN_EMBEDDING_API, headers=headers, json=payload)
    data = response.json()
    return [np.array(item["embedding"], dtype=np.float32) for item in data["output"]["embeddings"]]

# 构建FAISS索引
def build_faiss_index(embeddings: List[np.ndarray]) -> faiss.IndexFlatL2:
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    matrix = np.vstack(embeddings)
    index.add(matrix)
    return index

# 创建工具用于语义搜索
class FinanceSearchTool(Tool):
    def __init__(self, index: faiss.IndexFlatL2, docs: List[str], emb_fn):
        super().__init__(name="finance_search", description="基于金融新闻内容进行语义检索")
        self.index = index
        self.docs = docs
        self.emb_fn = emb_fn

    def call(self, query: str) -> str:
        emb = self.emb_fn([query])[0].reshape(1, -1)
        D, I = self.index.search(emb, k=3)
        return "\n".join([self.docs[i] for i in I[0]])

# 主流程
if __name__ == "__main__":
    print("正在抓取金融新闻数据……")
    docs = fetch_finance_news()
    print(f"已获取新闻{len(docs)}条，正在进行文本嵌入……")
    embeddings = get_embeddings(docs)
    print("构建向量数据库中……")
    index = build_faiss_index(embeddings)

    # 初始化语义检索工具
    search_tool = FinanceSearchTool(index=index, docs=docs, emb_fn=get_embeddings)

    # 初始化Qwen Agent
    agent = Agent(tools=[search_tool])

    # 构建对话消息
    user_msg = Message(role="user", content="最近有哪些关于货币政策的消息？")
    print("Agent正在生成回答……")
    response = agent.chat(messages=[user_msg])

    # 输出结果
    print("\n--- Agent回答 ---")
    print(response.content)




# 【例5-6】
import os
import json
import requests
import numpy as np
from typing import List
from qwen_agent.agent import Agent
from qwen_agent.tools import Tool
from qwen_agent.context import Message

# ========= 第一步：获取真实数据 =========
def fetch_energy_policy_titles() -> List[str]:
    url = "https://www.nea.gov.cn/api/front/index/policynews?pageIndex=1&pageSize=20"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    items = response.json().get("data", [])
    return [item["title"] for item in items if "title" in item]

# ========= 第二步：Embedding接口 =========
EMBEDDING_API = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
API_KEY = os.getenv("DASHSCOPE_API_KEY")

def get_qwen_embedding(texts: List[str]) -> List[np.ndarray]:
    payload = {
        "model": "text-embedding-v1",
        "input": texts
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    resp = requests.post(EMBEDDING_API, headers=headers, json=payload)
    data = resp.json()
    return [np.array(e["embedding"], dtype=np.float32) for e in data["output"]["embeddings"]]

# ========= 第三步：Faiss部署 =========
import faiss
class FaissSearchTool(Tool):
    def __init__(self, docs: List[str], embeddings: List[np.ndarray]):
        super().__init__(name="faiss_search", description="基于Faiss的政策语义搜索工具")
        self.docs = docs
        dim = embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.vstack(embeddings))

    def call(self, query: str) -> str:
        qvec = get_qwen_embedding([query])[0].reshape(1, -1)
        D, I = self.index.search(qvec, k=3)
        return "\n".join([self.docs[i] for i in I[0]])

# ========= 第四步：Chroma部署 =========
import chromadb
from chromadb.utils import embedding_functions

def build_chroma_client(docs: List[str], embeddings: List[np.ndarray]):
    client = chromadb.Client()
    collection = client.create_collection(name="energy_policy")
    for i, (doc, vec) in enumerate(zip(docs, embeddings)):
        collection.add(
            documents=[doc],
            ids=[f"doc{i}"],
            embeddings=[vec.tolist()]
        )
    return collection

class ChromaSearchTool(Tool):
    def __init__(self, collection):
        super().__init__(name="chroma_search", description="基于Chroma的政策语义搜索工具")
        self.collection = collection

    def call(self, query: str) -> str:
        emb = get_qwen_embedding([query])[0].tolist()
        result = self.collection.query(query_embeddings=[emb], n_results=3)
        return "\n".join(result["documents"][0])

# ========= 第五步：Weaviate部署 =========
import weaviate

def build_weaviate_client(docs: List[str], embeddings: List[np.ndarray]):
    client = weaviate.Client("http://localhost:8080")
    if not client.schema.exists("Policy"):
        client.schema.create_class({
            "class": "Policy",
            "vectorIndexType": "hnsw",
            "properties": [{"name": "content", "dataType": ["text"]}]
        })
    for i, (doc, vec) in enumerate(zip(docs, embeddings)):
        client.data_object.create(
            {"content": doc},
            "Policy",
            vector=vec.tolist()
        )
    return client

class WeaviateSearchTool(Tool):
    def __init__(self, client):
        super().__init__(name="weaviate_search", description="基于Weaviate的政策语义搜索工具")
        self.client = client

    def call(self, query: str) -> str:
        vec = get_qwen_embedding([query])[0].tolist()
        result = self.client.query.get("Policy", ["content"]).with_near_vector({"vector": vec}).with_limit(3).do()
        hits = result["data"]["Get"]["Policy"]
        return "\n".join([hit["content"] for hit in hits])

# ========= 第六步：执行检索 =========
if __name__ == "__main__":
    print("获取能源政策标题中……")
    docs = fetch_energy_policy_titles()
    print(f"获取到{len(docs)}条政策，开始编码……")
    embeddings = get_qwen_embedding(docs)

    # 构建三个向量数据库并封装工具
    faiss_tool = FaissSearchTool(docs, embeddings)
    chroma_tool = ChromaSearchTool(build_chroma_client(docs, embeddings))
    # weaviate_tool = WeaviateSearchTool(build_weaviate_client(docs, embeddings))  # 本地启动Weaviate后启用

    agent = Agent(tools=[faiss_tool, chroma_tool])  # 可添加 weaviate_tool

    query = "近期有哪些新能源相关政策？"
    print("Agent正在回答：", query)
    response = agent.chat(messages=[Message(role="user", content=query)])
    print("\n--- Agent回答 ---")
    print(response.content)




# 【例5-7】
import os
import json
import numpy as np
import faiss
import requests
from typing import List
from qwen_agent.agent import Agent
from qwen_agent.tools import Tool
from qwen_agent.context import Message

# ========= 第一步：加载法律文档 =========
def fetch_law_sections() -> List[str]:
    url = "https://www.lawxp.com/data/fake/law/ep_law.json"
    # 示例接口（请替换为真实链接或本地文件），格式为 [{"chapter": "...", "content": "..."}]
    response = requests.get(url)
    data = response.json()
    return [item["content"] for item in data if "content" in item]

# ========= 第二步：段落切分 =========
def split_text_into_chunks(texts: List[str], max_length=150) -> List[str]:
    chunks = []
    for text in texts:
        parts = text.split("。")
        chunk = ""
        for p in parts:
            if len(chunk) + len(p) < max_length:
                chunk += p + "。"
            else:
                chunks.append(chunk)
                chunk = p + "。"
        if chunk:
            chunks.append(chunk)
    return chunks

# ========= 第三步：Qwen嵌入接口 =========
EMBEDDING_API = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
API_KEY = os.getenv("DASHSCOPE_API_KEY")

def get_qwen_embedding(texts: List[str]) -> List[np.ndarray]:
    payload = {"model": "text-embedding-v1", "input": texts}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    resp = requests.post(EMBEDDING_API, headers=headers, json=payload)
    results = resp.json()["output"]["embeddings"]
    return [np.array(r["embedding"], dtype=np.float32) for r in results]

# ========= 第四步：构建向量索引 =========
class LawRetrievalTool(Tool):
    def __init__(self, chunks: List[str], embeddings: List[np.ndarray]):
        super().__init__(name="law_search", description="法律文本语义检索工具")
        self.chunks = chunks
        dim = embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.vstack(embeddings))

    def call(self, query: str) -> str:
        q_vec = get_qwen_embedding([query])[0].reshape(1, -1)
        D, I = self.index.search(q_vec, k=5)
        retrieved = [self.chunks[i] for i in I[0]]
        return "\n".join(retrieved)

# ========= 第五步：组装RetrievalQA上下文 =========
def build_prompt_with_context(query: str, context: str) -> str:
    prompt = f"""你是一名法律智能助手。请根据以下法律条文内容回答用户问题。
【法律条文】：
{context}
【用户提问】：
{query}
请结合条文内容给出明确、简洁、准确的回答。"""
    return prompt

# ========= 第六步：主流程 =========
if __name__ == "__main__":
    print("正在加载法律数据……")
    raw_sections = fetch_law_sections()
    print("原始条文数量：", len(raw_sections))
    
    print("正在切分文本为片段……")
    chunks = split_text_into_chunks(raw_sections)
    print("切分后片段数：", len(chunks))

    print("正在进行文本嵌入……")
    embeddings = get_qwen_embedding(chunks)

    # 初始化检索工具
    retrieval_tool = LawRetrievalTool(chunks, embeddings)

    # 初始化Agent
    agent = Agent(tools=[retrieval_tool])

    # 模拟用户提问
    question = "环境保护法中对企业排污行为有什么规定？"
    print("用户提问：", question)

    # 第一步：先调用工具进行语义检索
    context = retrieval_tool.call(question)

    # 第二步：拼接上下文并发送给Agent
    final_prompt = build_prompt_with_context(question, context)
    response = agent.chat(messages=[Message(role="user", content=final_prompt)])

    # 输出结果
    print("\n--- Agent回答 ---")
    print(response.content)




# 【例5-8】
import os
import json
import numpy as np
import faiss
import requests
from typing import List
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

# 第一步：加载真实政策数据（国家统计局官网公开API）
def fetch_policy_data() -> List[str]:
    url = "https://www.stats.gov.cn/sj/tjbz/tjzszc/index.json"  # 示例JSON格式公开接口
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    items = response.json().get("data", [])
    return [item["title"] for item in items if "title" in item]

# 第二步：段落切分策略
def split_chunks(texts: List[str], max_len=100) -> List[str]:
    chunks = []
    for text in texts:
        if len(text) <= max_len:
            chunks.append(text)
        else:
            chunks.extend([text[i:i + max_len] for i in range(0, len(text), max_len)])
    return chunks

# 第三步：Qwen嵌入接口（text-embedding-v1）
EMBEDDING_API = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
API_KEY = os.getenv("DASHSCOPE_API_KEY")

def get_qwen_embedding(texts: List[str]) -> List[np.ndarray]:
    payload = {"model": "text-embedding-v1", "input": texts}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    response = requests.post(EMBEDDING_API, headers=headers, json=payload)
    results = response.json()["output"]["embeddings"]
    return [np.array(r["embedding"], dtype=np.float32) for r in results]

# 第四步：构建Top-K融合的多段检索工具
class MultiPassageRetrievalTool(Tool):
    def __init__(self, docs: List[str], embeddings: List[np.ndarray], top_k: int = 5):
        super().__init__(name="multi_passage_search", description="多段语义检索与Top-K融合工具")
        self.docs = docs
        self.top_k = top_k
        dim = embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.vstack(embeddings))

    def call(self, query: str) -> str:
        qvec = get_qwen_embedding([query])[0].reshape(1, -1)
        D, I = self.index.search(qvec, self.top_k)
        top_passages = [self.docs[i] for i in I[0]]
        # Top-K融合策略：拼接多个片段形成更丰富上下文
        fused_context = "\n".join(top_passages)
        return fused_context

# 第五步：构建Agent问答上下文
def build_prompt(query: str, context: str) -> str:
    return f"""你是一名智能问答助手，以下是从政策数据库中检索到的多段内容：
【语料片段】
{context}

请基于上述内容准确回答以下问题：
【提问】{query}
请用中文简洁清晰地作答。"""

# 第六步：执行主流程
if __name__ == "__main__":
    print("正在加载政策语料数据……")
    raw_docs = fetch_policy_data()
    print(f"原始政策标题数：{len(raw_docs)}")

    print("正在进行分段处理……")
    chunks = split_chunks(raw_docs)
    print(f"分段后共 {len(chunks)} 段")

    print("开始生成嵌入向量……")
    embeddings = get_qwen_embedding(chunks)

    print("构建向量数据库并集成Agent……")
    tool = MultiPassageRetrievalTool(chunks, embeddings, top_k=5)
    agent = Agent(tools=[tool])

    # 用户问题输入
    query = "国家统计局有哪些针对GDP核算方法的文件或解释？"
    print("用户提问：", query)
    # 第一步：检索多段内容
    retrieved = tool.call(query)
    # 第二步：构造最终对话上下文
    prompt = build_prompt(query, retrieved)
    # 第三步：调用Agent生成回答
    response = agent.chat(messages=[Message(role="user", content=prompt)])

    print("\n--- Agent回答 ---")
    print(response.content)




# 【例5-9】
import os
import json
import requests
import pandas as pd
import numpy as np
import faiss
from typing import List, Dict
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

# ========= 第一步：加载真实文本和结构化数据 =========
def fetch_fiscal_texts() -> List[str]:
    url = "https://www.mof.gov.cn/zhengwuxinxi/caizhengxinwen/index.json"  # 模拟财政新闻JSON接口
    resp = requests.get(url)
    data = resp.json().get("news", [])
    return [item["title"] + "。" + item.get("summary", "") for item in data[:20]]

def fetch_structured_data() -> pd.DataFrame:
    # 模拟真实财政年度数据CSV
    url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Government%20revenue%20and%20spending/Government_revenue_and_spending.csv"
    df = pd.read_csv(url)
    df = df[df["Entity"] == "China"]
    return df[["Year", "Government spending (as % of GDP)", "Government revenue (as % of GDP)"]].dropna()

# ========= 第二步：Qwen嵌入接口 =========
EMBEDDING_API = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
API_KEY = os.getenv("DASHSCOPE_API_KEY")

def get_qwen_embedding(texts: List[str]) -> List[np.ndarray]:
    payload = {"model": "text-embedding-v1", "input": texts}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    resp = requests.post(EMBEDDING_API, headers=headers, json=payload)
    data = resp.json()["output"]["embeddings"]
    return [np.array(e["embedding"], dtype=np.float32) for e in data]

# ========= 第三步：文本检索工具 =========
class FiscalTextSearchTool(Tool):
    def __init__(self, docs: List[str], embeddings: List[np.ndarray]):
        super().__init__(name="fiscal_text_search", description="财政政策文本语义搜索")
        self.docs = docs
        dim = embeddings[0].shape[0]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(np.vstack(embeddings))

    def call(self, query: str) -> str:
        vec = get_qwen_embedding([query])[0].reshape(1, -1)
        D, I = self.index.search(vec, k=3)
        return "\n".join([self.docs[i] for i in I[0]])

# ========= 第四步：结构化数据查询工具 =========
class FiscalDataQueryTool(Tool):
    def __init__(self, df: pd.DataFrame):
        super().__init__(name="fiscal_data_query", description="财政数据结构化查询工具")
        self.df = df

    def call(self, query: str) -> str:
        # 使用DeepSeek-V1进行自然语言转SQL（模拟）
        if "2020" in query and "支出" in query:
            row = self.df[self.df["Year"] == 2020]
            return f"2020年中国政府支出占GDP比重为{row['Government spending (as % of GDP)'].values[0]}%。"
        elif "收入" in query:
            year = 2021
            row = self.df[self.df["Year"] == year]
            return f"{year}年中国政府收入占GDP比重为{row['Government revenue (as % of GDP)'].values[0]}%。"
        else:
            return "未能识别查询字段，请重新表述问题。"

# ========= 第五步：组装Agent响应流程 =========
def build_prompt(query: str, text_context: str, data_context: str) -> str:
    return f"""你是一个财税智能助手，拥有文本政策知识和财政结构化数据。
【政策解读】：
{text_context}
【财政数据】：
{data_context}
请结合以上信息回答用户问题：
{query}
回答应准确、简洁、有数据支撑。"""

# ========= 第六步：执行主流程 =========
if __name__ == "__main__":
    print("加载财政文本与结构化数据……")
    texts = fetch_fiscal_texts()
    df = fetch_structured_data()
    print("加载完成，共加载文本%d条，结构化数据%d行。" % (len(texts), len(df)))

    print("生成文本嵌入向量中……")
    text_embeddings = get_qwen_embedding(texts)

    # 初始化工具
    text_tool = FiscalTextSearchTool(texts, text_embeddings)
    data_tool = FiscalDataQueryTool(df)

    # 初始化Agent
    agent = Agent(tools=[text_tool, data_tool])

    # 用户问题
    query = "2020年中国的财政支出是多少？相关政策有没有调整说明？"
    print("用户提问：", query)

    # 第一步：文本工具检索政策解读
    text_context = text_tool.call(query)

    # 第二步：结构化数据工具查询
    data_context = data_tool.call(query)

    # 第三步：组合提示词并生成
    full_prompt = build_prompt(query, text_context, data_context)
    response = agent.chat(messages=[Message(role="user", content=full_prompt)])

    print("\n--- Agent回答 ---")
    print(response.content)
