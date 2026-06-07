# 【例4-7】
# 4_7_book_search_memory_agent.py
#
# 本例演示图书搜索与跨轮记忆：
# 1. OpenLibrary搜索图书列表
# 2. 记住上一轮搜索结果
# 3. 用户追问第一本书时查询详情并让LLM中文介绍
#
# 本地Ollama运行：
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch04/src/4_7_book_search_memory_agent.py
#
# 云端DeepSeek/OpenAI兼容API运行：
# python3 ch04/src/4_7_book_search_memory_agent.py

import requests

from llm_backend import ask_text, backend_name


# 工具1：基于关键词搜索图书
def search_books(keyword: str) -> list[dict]:
    """调用公开API搜索图书标题"""
    url = f"https://openlibrary.org/search.json?q={keyword}"
    res = requests.get(url, timeout=10)
    return res.json().get("docs", [])[:3]


def format_books(docs: list[dict]) -> str:
    return "\n".join(
        f"{i + 1}. {doc.get('title')} by {doc.get('author_name', ['N/A'])[0]}"
        for i, doc in enumerate(docs)
    )


# 工具2：根据书名获取图书详细信息
def get_book_details(title: str) -> str:
    """根据书名获取图书出版详情"""
    url = f"https://openlibrary.org/search.json?title={title}"
    res = requests.get(url, timeout=10)
    docs = res.json().get("docs", [])
    if not docs:
        return "未找到该书籍详情"
    book = docs[0]
    return (
        f"书名: {book.get('title')}\n"
        f"作者: {book.get('author_name', ['N/A'])[0]}\n"
        f"出版时间: {book.get('first_publish_year', '未知')}"
    )


class BookMemory:
    def __init__(self):
        self.last_books: list[dict] = []


def run_agent(user_input: str, memory: BookMemory) -> str:
    if "第一本" in user_input and memory.last_books:
        title = memory.last_books[0].get("title", "")
        details = get_book_details(title)
        return ask_text(
            f"请用中文简要介绍这本书：\n{details}",
            system="你是一个图书推荐助手。回答不超过180字，不要输出思考过程。",
            temperature=0,
            max_tokens=300,
        )

    keyword = "人工智能" if "人工智能" in user_input else user_input
    memory.last_books = search_books(keyword)
    return format_books(memory.last_books)


def main():
    print(f"LLM后端：{backend_name()}\n")
    memory = BookMemory()

    # 多轮调用：先搜索，再根据第一本书名查询详情
    response_1 = run_agent("我想查几本关于人工智能的书", memory)
    response_2 = run_agent("请介绍第一本书的详细信息", memory)

    # 输出结果
    print(response_1)
    print(response_2)


if __name__ == "__main__":
    main()
