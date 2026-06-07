# 第4章 LangChain框架与智能体构建流程
# 【例4-1】
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

# 初始化大语言模型
llm = OpenAI(temperature=0.5)

# 子链一：根据主题生成子主题
prompt1 = PromptTemplate(
    input_variables=["topic"],
    template="请列出与“{topic}”相关的三个细分研究方向，用逗号分隔。"
)
chain1 = LLMChain(llm=llm, prompt=prompt1, output_key="subtopics")

# 子链二：基于子主题生成研究问题
prompt2 = PromptTemplate(
    input_variables=["subtopics"],
    template="以下是子主题：{subtopics}，请为每个子主题提出一个具挑战性的研究问题。"
)
chain2 = LLMChain(llm=llm, prompt=prompt2, output_key="questions")

# 子链三：总结研究问题的社会价值
prompt3 = PromptTemplate(
    input_variables=["questions"],
    template="以下是若干研究问题：{questions}，请简要分析其对于推动社会发展的意义。"
)
chain3 = LLMChain(llm=llm, prompt=prompt3, output_key="impact")

# 构造完整链式结构
full_chain = SequentialChain(
    chains=[chain1, chain2, chain3],
    input_variables=["topic"],
    output_variables=["subtopics", "questions", "impact"],
    verbose=True
)

# 执行任务
response = full_chain.run(topic="人工智能伦理")

# 打印输出
print(response)




# 【例4-2】
from langchain.agents import Tool, initialize_agent
from langchain.agents.agent_types import AgentType
from langchain.llms import OpenAI
import requests

# 工具1：网页内容摘要
def fetch_summary(url: str) -> str:
    try:
        resp = requests.get(url, timeout=10)
        if resp.status_code == 200:
            content = resp.text[:1000]
            return f"网页摘要：{content[:300]}..."  # 仅返回前300字
        else:
            return "无法访问网页内容"
    except Exception as e:
        return str(e)

# 工具2：美元转人民币汇率计算
def convert_usd_to_cny(amount: float) -> str:
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

# 初始化LLM和Agent
llm = OpenAI(temperature=0.3)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# 测试任务
response = agent.run("请帮我把20美元换算成人民币")

print("==== 输出结果 ====")
print(response)




# 【例4-3】
from langchain.agents import initialize_agent, Tool
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

# 自定义工具：天气响应模拟
def get_weather(city: str) -> str:
    dummy_weather = {
        "北京": "晴，25°C",
        "上海": "多云，23°C",
        "广州": "雷阵雨，29°C"
    }
    return dummy_weather.get(city, "当前城市暂无天气数据")

weather_tool = Tool(
    name="WeatherTool",
    func=get_weather,
    description="查询城市天气，如输入：北京"
)

# 初始化Memory：记录用户每轮对话
memory = ConversationBufferMemory(
    memory_key="chat_history",
    input_key="input"
)

# 初始化模型与Agent
llm = OpenAI(temperature=0.4)
agent = initialize_agent(
    tools=[weather_tool],
    llm=llm,
    memory=memory,
    agent="zero-shot-react-description",
    verbose=True
)

# 连续执行多轮对话任务
print("用户：我想知道北京天气如何？")
res1 = agent.run("我想知道北京天气如何？")
print("回答：", res1)

print("\n用户：那广州呢？")
res2 = agent.run("那广州呢？")
print("回答：", res2)

print("\n用户：那明天去哪旅游好？")
res3 = agent.run("那明天去哪旅游好？")
print("回答：", res3)




# 【例4-4】
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
import time
import signal

# 限时执行器定义
class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("执行超时")

signal.signal(signal.SIGALRM, timeout_handler)

# 安全计算表达式的沙箱函数
def safe_eval(expression: str) -> str:
    signal.alarm(3)  # 限制最多3秒
    try:
        # 允许的表达式类型限制
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return "非法表达式，包含禁止字符"
        result = eval(expression, {"__builtins__": {}})
        return f"结果为：{result}"
    except TimeoutException:
        return "执行超时"
    except Exception as e:
        return f"执行失败：{str(e)}"
    finally:
        signal.alarm(0)

# 工具定义
safe_calc_tool = Tool(
    name="SafeCalculator",
    func=safe_eval,
    description="用于安全计算加减乘除表达式，如：3*(5+2)"
)

# 初始化Agent
memory = ConversationBufferMemory()
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools=[safe_calc_tool],
    llm=llm,
    memory=memory,
    agent="zero-shot-react-description",
    verbose=True
)

# 多轮调用验证沙箱效果
print("用户输入：计算表达式 3*(5+2)")
res1 = agent.run("计算表达式 3*(5+2)")
print("模型输出：", res1)

print("\n用户输入：执行表达式 import os; os.system('rm -rf /')")
res2 = agent.run("执行表达式 import os; os.system('rm -rf /')")
print("模型输出：", res2)




# 【例4-5】
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.tools import tool
from typing import List
import requests

# 工具1：网络搜索接口（模拟）
@tool
def search_web(query: str) -> str:
    """根据用户输入执行Web搜索，并返回摘要结果"""
    if "AI Agent" in query:
        return "AI Agent是一种基于大语言模型的自主任务执行单元，具备工具调用与上下文感知能力。"
    return f"未找到关于：{query} 的相关信息"

# 工具2：摘要生成器
@tool
def summarize_text(text: str) -> str:
    """对给定文本执行摘要处理"""
    return "智能体是一种基于大模型的自主系统。"

# 工具3：翻译模块
@tool
def translate_to_zh(text: str) -> str:
    """将英文摘要翻译成中文"""
    return "智能体是一种基于大模型的自主系统。"

# 工具集成
tools = [search_web, summarize_text, translate_to_zh]

# 初始化模型与Agent
llm = OpenAI(temperature=0)
memory = ConversationBufferMemory()
agent = initialize_agent(
    tools=tools,
    llm=llm,
    memory=memory,
    agent="zero-shot-react-description",
    verbose=True
)

# 任务请求：用中文总结AI Agent的基本定义
print("用户输入：用中文总结AI Agent的基本定义")
output = agent.run("用中文总结AI Agent的基本定义")
print("模型输出：", output)




# 【例4-6】
# 引入基础组件
from langchain.chat_models import ChatOpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate
import requests

# 工具函数：天气查询
def get_weather(city: str) -> str:
    try:
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url)
        return response.text.strip()
    except Exception as e:
        return f"查询失败：{str(e)}"

# 工具函数：城市百科简介（调用维基百科API）
def get_city_intro(city: str) -> str:
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{city}"
        response = requests.get(url).json()
        return response.get("extract", "未找到该城市简介")
    except Exception as e:
        return f"查询失败：{str(e)}"

# 注册工具：需说明name, func, description
tools = [
    Tool(
        name="GetWeather",
        func=get_weather,
        description="输入城市名称，返回该城市当前天气情况"
    ),
    Tool(
        name="GetCityIntro",
        func=get_city_intro,
        description="输入城市名称，返回该城市的百科简介"
    )
]

# 自定义Prompt模板
custom_prompt = PromptTemplate(
    input_variables=["input", "history"],
    template="""
你是一位知识渊博的智能助理，可以调用两种工具：GetWeather用于查询天气，GetCityIntro用于提供城市简介。
用户的问题如下：{input}
历史对话记录如下：{history}
请根据问题内容选择合适的工具进行调用，并用自然语言回答。
"""
)

# 配置对话记忆
memory = ConversationBufferMemory(memory_key="history")

# 初始化大语言模型
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.5)

# 构建Agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True,
    agent_kwargs={"prompt": custom_prompt}
)

# 启动Agent并模拟多轮对话
print("====== 实际对话示例 ======")
print(agent.run("介绍一下上海"))
print(agent.run("那现在上海的天气如何？"))
print(agent.run("北京天气怎么样？"))




# 【例4-7】
import os
import requests
from langchain.chat_models import ChatOpenAI
from langchain.agents import Tool, initialize_agent
from langchain.memory import ConversationBufferMemory
from langchain.agents.agent_types import AgentType
# 工具1：基于关键词搜索图书
def search_books(keyword: str) -> str:
    """调用公开API搜索图书标题"""
    url = f"https://openlibrary.org/search.json?q={keyword}"
    res = requests.get(url)
    docs = res.json().get("docs", [])[:3]
    return "\n".join([f"{i+1}. {doc.get('title')} by {doc.get('author_name', ['N/A'])[0]}" for i, doc in enumerate(docs)])
# 工具2：根据书名获取图书详细信息
def get_book_details(title: str) -> str:
    """根据书名获取图书出版详情"""
    url = f"https://openlibrary.org/search.json?title={title}"
    res = requests.get(url)
    docs = res.json().get("docs", [])
    if not docs:
        return "未找到该书籍详情"
    book = docs[0]
    return f"书名: {book.get('title')}\n作者: {book.get('author_name', ['N/A'])[0]}\n出版时间: {book.get('first_publish_year', '未知')}"
# 定义工具集
tools = [
    Tool(
        name="SearchBooks",
        func=search_books,
        description="根据关键词搜索图书，适合用于确定书名"
    ),
    Tool(
        name="GetBookDetails",
        func=get_book_details,
        description="根据书名获取图书详细信息"
    )
]
# 初始化语言模型与Agent
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
memory = ConversationBufferMemory(memory_key="history")
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
    memory=memory,
    verbose=True
)
# 多轮调用：先搜索，再根据第一本书名查询详情
response_1 = agent.run("我想查几本关于人工智能的书")
response_2 = agent.run("请介绍第一本书的详细信息")
# 输出结果
print(response_1)
print(response_2)
