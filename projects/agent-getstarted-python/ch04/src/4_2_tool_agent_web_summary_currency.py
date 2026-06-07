# 【例4-2】
# 4_2_tool_agent_web_summary_currency.py
#
# 本例演示一个轻量工具型Agent：
# 1. URL输入走网页正文提取和摘要工具
# 2. 美元金额输入走汇率换算工具
# 3. 其他问题交给LLM直接回答
#
# 本地Ollama运行：
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch04/src/4_2_tool_agent_web_summary_currency.py
#
# 云端DeepSeek/OpenAI兼容API运行：
# python3 ch04/src/4_2_tool_agent_web_summary_currency.py

import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Callable

import requests

from llm_backend import ask_text, backend_name


@dataclass
class Tool:
    name: str
    func: Callable
    description: str


class WebTextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.title = ""
        self.description = ""
        self._in_title = False
        self._skip_depth = 0
        self._parts: list[str] = []

    def handle_starttag(self, tag: str, attrs):
        attrs = dict(attrs)
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"}:
            self._skip_depth += 1
        elif tag == "title":
            self._in_title = True
        elif tag == "meta" and attrs.get("name", "").lower() == "description":
            self.description = attrs.get("content", "")

    def handle_endtag(self, tag: str):
        tag = tag.lower()
        if tag in {"script", "style", "noscript", "svg"} and self._skip_depth:
            self._skip_depth -= 1
        elif tag == "title":
            self._in_title = False

    def handle_data(self, data: str):
        text = data.strip()
        if not text:
            return
        if self._in_title:
            self.title += text
        elif not self._skip_depth:
            self._parts.append(text)

    def readable_text(self) -> str:
        raw = "\n".join([self.title, self.description, *self._parts])
        text = unescape(raw)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


def extract_readable_text(html: str) -> str:
    parser = WebTextExtractor()
    parser.feed(html)
    return parser.readable_text()


# 工具1：网页内容摘要
def fetch_summary(url: str) -> str:
    try:
        resp = requests.get(
            url,
            timeout=10,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/125.0 Safari/537.36"
                )
            },
        )
        if resp.status_code != 200:
            return "无法访问网页内容"

        encoding = resp.encoding
        if not encoding or encoding.lower() == "iso-8859-1":
            encoding = resp.apparent_encoding or "utf-8"
        html = resp.content.decode(encoding, errors="replace")
        text = extract_readable_text(html)
        if not text:
            return "网页可访问，但未提取到可读正文内容"

        prompt = (
            "请用中文概括下面网页内容，要求不超过120字，"
            "不要输出HTML标签，也不要逐字复制导航菜单。\n\n"
            f"网页URL：{url}\n"
            f"网页文本：{text[:3000]}"
        )
        try:
            return "网页摘要：" + ask_llm(prompt)
        except Exception:
            return "网页摘要：" + text[:300] + "..."
    except Exception as e:
        return str(e)


# 工具2：美元转人民币汇率计算
def convert_usd_to_cny(amount: float | str) -> str:
    amount = float(amount)
    rate = 7.25
    converted = round(amount * rate, 2)
    return f"{amount}美元 ≈ {converted}人民币（按汇率7.25）"

# 包装为LangChain工具
tools = [
    Tool(
        name="WebSummaryTool",
        func=fetch_summary,
        description="根据提供的网址提取网页摘要，适用于阅读网页内容",
    ),
    Tool(
        name="USDToCNYConverter",
        func=convert_usd_to_cny,
        description="将美元金额转换成人民币金额，适用于财务相关查询",
    )
]


def ask_llm(user_input: str) -> str:
    return ask_text(
        user_input,
        system="你是一个简洁可靠的中文工具助手。回答不要输出思考过程。",
        temperature=0.3,
        max_tokens=512,
    )


def run_agent(user_input: str) -> str:
    url_match = re.search(r"https?://\S+", user_input)
    if url_match:
        return fetch_summary(url_match.group(0))

    amount_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:美元|美金|usd|USD)", user_input)
    if amount_match and ("人民币" in user_input or "换算" in user_input or "转换" in user_input):
        return convert_usd_to_cny(amount_match.group(1))

    tool_descriptions = "\n".join(f"- {tool.name}: {tool.description}" for tool in tools)
    return ask_llm(
        "你是一个工具型助手。当前可用工具如下：\n"
        f"{tool_descriptions}\n\n"
        f"用户问题：{user_input}\n"
        "如果无需调用工具，请直接回答。"
    )


def main():
    print(f"LLM后端：{backend_name()}\n")

    # 测试任务1：网页摘要
    response1 = run_agent("请帮我总结一下这个网页的内容：https://www.vmall.com/")
    print("==== 网页摘要结果 ====")
    print(response1)

    # 测试任务2：货币转换
    response2 = run_agent("请帮我把20美元换算成人民币")
    print("\n==== 货币转换结果 ====")
    print(response2)


if __name__ == "__main__":
    main()
