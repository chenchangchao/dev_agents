# 第12章 项目案例：从零实现一个复合智能体系统
# 【例12-1】
# -*- coding:utf-8 -*-
# 多模型Agent的数据结构与接口定义模块

import uuid
import traceback
from typing import List, Dict, Any
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# 一、统一上下文结构定义
class Message:
    def __init__(self, role: str, content: str):
        self.role = role
        self.content = content

    def to_langchain(self):
        if self.role == "user":
            return HumanMessage(content=self.content)
        elif self.role == "assistant":
            return AIMessage(content=self.content)
        elif self.role == "system":
            return SystemMessage(content=self.content)
        else:
            raise ValueError(f"不支持的角色类型：{self.role}")

# 二、标准化上下文管理
class ConversationContext:
    def __init__(self, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.messages: List[Message] = []

    def add_message(self, role, content):
        self.messages.append(Message(role, content))

    def get_langchain_messages(self):
        return [msg.to_langchain() for msg in self.messages]

    def as_dict(self):
        return [{"role": msg.role, "content": msg.content} for msg in self.messages]

# 三、模型适配器封装
class ModelAdapter:
    def __init__(self, model_type="qwen"):
        self.model_type = model_type
        self.model = self._load_model(model_type)

    def _load_model(self, model_type):
        if model_type == "qwen":
            return Qwen2Chat(model="Qwen/Qwen1.5-7B-Chat")
        elif model_type == "deepseek":
            return DeepSeekChat(model="deepseek-ai/deepseek-llm-chat")
        else:
            raise ValueError("不支持的模型类型")

    def call(self, context: ConversationContext):
        try:
            messages = context.get_langchain_messages()
            response = self.model(messages)
            return {
                "success": True,
                "response": response.content
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc()
            }

# 四、实际应用示例
def run_demo():
    print("=== 示例：使用Qwen模型进行对话 ===")
    context = ConversationContext()
    context.add_message("system", "你是一个专业的AI助手")
    context.add_message("user", "请简要解释什么是注意力机制")

    adapter = ModelAdapter(model_type="qwen")
    result = adapter.call(context)
    if result["success"]:
        print("[Qwen 回复]", result["response"])
        context.add_message("assistant", result["response"])
    else:
        print("[Qwen 失败]", result["error"])

    print("\n=== 使用Deepseek模型继续对话 ===")
    context.add_message("user", "它和多头注意力有什么关系？")
    adapter2 = ModelAdapter(model_type="deepseek")
    result2 = adapter2.call(context)
    if result2["success"]:
        print("[Deepseek 回复]", result2["response"])
    else:
        print("[Deepseek 失败]", result2["error"])

run_demo()




# 【例12-2】
# -*- coding:utf-8 -*-
# 用户意图识别与入口解析模块，支持多模型调用与结果结构化

import uuid
import traceback
from typing import Dict, Any
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain.schema import HumanMessage

# 一、意图识别模板定义
INTENT_TEMPLATE = """
请将以下用户请求进行意图识别，返回格式为JSON，包含：
- intent：意图类别（如 "查询", "执行", "对话", "工具调用"）
- domain：任务领域（如 "天气", "数据库", "文本生成"）
- parameters：提取的关键词参数（以键值对形式返回）

用户请求：
"{query}"
"""

# 二、模型调用器
class ModelInvoker:
    def __init__(self, model_type="qwen"):
        self.model_type = model_type
        self.model = self._load_model()

    def _load_model(self):
        if self.model_type == "qwen":
            return Qwen2Chat(model="Qwen/Qwen1.5-7B-Chat")
        elif self.model_type == "deepseek":
            return DeepSeekChat(model="deepseek-ai/deepseek-llm-chat")
        else:
            raise ValueError("不支持的模型类型")

    def run_intent_recognition(self, user_query: str) -> Dict[str, Any]:
        prompt = INTENT_TEMPLATE.replace("{query}", user_query)
        message = [HumanMessage(content=prompt)]
        try:
            response = self.model(message)
            result = eval(response.content.strip())
            return {"success": True, "data": result}
        except Exception as e:
            return {"success": False, "error": str(e), "traceback": traceback.format_exc()}

# 三、意图识别模块，支持多模型兜底
class IntentParser:
    def __init__(self):
        self.primary = ModelInvoker("qwen")
        self.backup = ModelInvoker("deepseek")

    def parse(self, query: str) -> Dict[str, Any]:
        result = self.primary.run_intent_recognition(query)
        if result["success"]:
            return result["data"]
        print("[WARN] Qwen解析失败，尝试使用Deepseek")
        result = self.backup.run_intent_recognition(query)
        if result["success"]:
            return result["data"]
        print("[ERROR] 所有模型解析失败")
        return {"intent": "未知", "domain": "未知", "parameters": {}}

# 四、模拟实际调用流程
def run_demo():
    parser = IntentParser()

    queries = [
        "请帮我查一下今天上海的天气",
        "生成一篇关于人工智能未来发展的短文",
        "将这段文字翻译成英文：你好世界",
        "帮我执行一下数据库清理脚本",
        "你好，你能陪我聊聊天吗？"
    ]

    for i, q in enumerate(queries):
        print(f"\n用户请求{i+1}：{q}")
        result = parser.parse(q)
        print("识别结果：", result)

run_demo()




# 【例12-3】
# -*- coding:utf-8 -*-
# 工具链调用与回退机制实现（Qwen3.0 + Deepseek + LangChain Tool）

import traceback
from typing import List
from langchain_core.tools import tool
from langchain.agents import initialize_agent, AgentType
from langchain.agents import Tool
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain.schema import HumanMessage

# 一、定义可供调用的工具函数
@tool
def query_weather(city: str) -> str:
    """查询某地当前天气（模拟接口）"""
    if city.lower() == "未知":
        raise Exception("无法获取指定城市天气信息")
    return f"{city}当前天气：晴，气温26°C"

@tool
def calculate_expression(expr: str) -> str:
    """计算表达式结果"""
    try:
        result = eval(expr)
        return f"计算结果为：{result}"
    except:
        return "表达式有误，无法计算"

TOOLS = [
    Tool.from_function(query_weather),
    Tool.from_function(calculate_expression),
]

# 二、工具Agent管理器
class ToolAgentManager:
    def __init__(self):
        self.primary_model = Qwen2Chat(model="Qwen/Qwen1.5-7B-Chat")
        self.backup_model = DeepSeekChat(model="deepseek-ai/deepseek-llm-chat")

    def run_with_tools(self, input_text: str) -> str:
        # 首先尝试Qwen模型
        try:
            print("[INFO] 使用Qwen执行任务")
            agent = initialize_agent(
                TOOLS,
                self.primary_model,
                agent=AgentType.OPENAI_FUNCTIONS,
                verbose=False,
            )
            return agent.run(input_text)
        except Exception as e:
            print("[WARN] Qwen执行失败，回退至Deepseek")
            print(traceback.format_exc())
            # 若Qwen失败，尝试Deepseek
            try:
                agent = initialize_agent(
                    TOOLS,
                    self.backup_model,
                    agent=AgentType.OPENAI_FUNCTIONS,
                    verbose=False,
                )
                return agent.run(input_text)
            except Exception as e2:
                print("[ERROR] 所有模型调用失败")
                print(traceback.format_exc())
                return "调用失败，系统暂时无法完成该请求。"

# 三、模拟调用示例
def run_demo():
    manager = ToolAgentManager()

    inputs = [
        "帮我查一下北京的天气",
        "计算 7 * (8 + 3)",
        "请查一下未知城市的天气",
        "将这段话翻译为英文：你好",
    ]

    for i, text in enumerate(inputs):
        print(f"\n输入{i+1}：{text}")
        result = manager.run_with_tools(text)
        print("输出结果：", result)

run_demo()




# 【例12-4】
# -*- coding:utf-8 -*-
# Agent子系统状态管理与调度机制

import time
import uuid
import traceback
from typing import Dict, List
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain.schema import HumanMessage

# 一、子Agent状态定义
class AgentState:
    def __init__(self, name: str):
        self.name = name
        self.status = "idle"  # idle, running, success, failed
        self.last_result = None
        self.last_error = None

    def update(self, status: str, result=None, error=None):
        self.status = status
        self.last_result = result
        self.last_error = error

# 二、任务调度控制器
class AgentController:
    def __init__(self):
        self.session_id = str(uuid.uuid4())
        self.agents: Dict[str, AgentState] = {
            "intent": AgentState("意图识别Agent"),
            "qa": AgentState("问答Agent"),
            "tool": AgentState("工具调用Agent"),
            "dialog": AgentState("对话Agent"),
        }
        self.qwen = Qwen2Chat(model="Qwen/Qwen1.5-7B-Chat")
        self.deepseek = DeepSeekChat(model="deepseek-ai/deepseek-llm-chat")

    def _run_model(self, model, prompt: str) -> str:
        try:
            messages = [HumanMessage(content=prompt)]
            response = model(messages)
            return response.content.strip()
        except Exception:
            raise

    def _execute_with_fallback(self, prompt: str) -> str:
        try:
            return self._run_model(self.qwen, prompt)
        except Exception as e:
            print("[WARN] Qwen执行失败，切换至Deepseek")
            try:
                return self._run_model(self.deepseek, prompt)
            except Exception:
                raise RuntimeError("所有模型均执行失败")

    def dispatch_task(self, task_type: str, input_text: str) -> str:
        agent = self.agents.get(task_type)
        if not agent:
            return f"未找到任务类型：{task_type}"
        agent.update("running")
        try:
            if task_type == "intent":
                prompt = f"请识别该用户输入的意图：{input_text}"
            elif task_type == "qa":
                prompt = f"请回答该问题：{input_text}"
            elif task_type == "tool":
                prompt = f"是否需要调用工具来完成该任务？{input_text}"
            elif task_type == "dialog":
                prompt = f"请以自然语言与用户继续交谈：{input_text}"
            else:
                raise ValueError("未知任务类型")
            result = self._execute_with_fallback(prompt)
            agent.update("success", result=result)
            return result
        except Exception as e:
            agent.update("failed", error=str(e))
            return f"[ERROR] {agent.name}执行失败：{str(e)}"

    def get_status_snapshot(self) -> Dict[str, str]:
        return {k: v.status for k, v in self.agents.items()}

# 三、模拟调度流程
def run_demo():
    controller = AgentController()

    tasks = [
        ("intent", "我想查一下北京的天气"),
        ("tool", "现在外面温度是多少？"),
        ("qa", "Transformer的注意力机制原理是什么？"),
        ("dialog", "你好，可以和我聊聊人工智能吗？"),
    ]

    for i, (task_type, text) in enumerate(tasks):
        print(f"\n任务{i+1}：{task_type} → {text}")
        output = controller.dispatch_task(task_type, text)
        print("任务输出：", output)

    print("\n【Agent状态快照】")
    print(controller.get_status_snapshot())

run_demo()




# 【例12-5】
# -*- coding:utf-8 -*-
# RAG检索子系统实现：向量检索+模型生成

import os
import uuid
import traceback
from langchain.chains import ConversationalRetrievalChain
from langchain.vectorstores import Chroma
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain.schema import HumanMessage

# 一、准备向量存储（文档→嵌入→Chroma）
def build_vectorstore_from_file(file_path: str, persist_dir="./chroma_rag"):
    loader = TextLoader(file_path, encoding='utf-8')
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = splitter.split_documents(docs)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(split_docs, embedding=embedding_model, persist_directory=persist_dir)
    vectorstore.persist()
    return vectorstore

# 二、构建RAG问答系统
class RAGAgent:
    def __init__(self, vectorstore, model_type="qwen"):
        self.vectorstore = vectorstore
        self.model_type = model_type
        self.model = self._load_model()
        self.memory = ConversationBufferMemory(return_messages=True)

    def _load_model(self):
        if self.model_type == "qwen":
            return Qwen2Chat(model="Qwen/Qwen1.5-7B-Chat")
        else:
            return DeepSeekChat(model="deepseek-ai/deepseek-llm-chat")

    def ask(self, query: str):
        try:
            chain = ConversationalRetrievalChain.from_llm(
                llm=self.model,
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": 2}),
                memory=self.memory
            )
            result = chain.run(query)
            return result
        except Exception as e:
            return f"[ERROR] 回答失败：{str(e)}"

# 三、测试运行：准备数据并调用RAG系统
def run_demo():
    print(">>> 正在构建向量知识库...")
    vectorstore = build_vectorstore_from_file("data/ai_knowledge.txt")

    print(">>> 初始化RAG智能体")
    agent = RAGAgent(vectorstore, model_type="qwen")

    queries = [
        "请简要介绍什么是人工智能？",
        "人工智能和机器学习有什么区别？",
        "深度学习属于人工智能吗？",
    ]

    for i, q in enumerate(queries):
        print(f"\n问题{i+1}：{q}")
        reply = agent.ask(q)
        print("回答：", reply)

run_demo()




# 【例12-6】
# -*- coding:utf-8 -*-
# MCP上下文路由配置与调用系统

import uuid
from typing import List, Dict
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain.schema import HumanMessage, AIMessage, SystemMessage

# 一、定义MCP消息协议结构
class MCPMessage:
    def __init__(self, role: str, content: str, call_type: str = "prompt"):
        self.role = role  # user, assistant, system
        self.content = content
        self.call_type = call_type  # prompt, tool_call, response

    def to_langchain(self):
        if self.role == "user":
            return HumanMessage(content=self.content)
        elif self.role == "assistant":
            return AIMessage(content=self.content)
        elif self.role == "system":
            return SystemMessage(content=self.content)
        else:
            raise ValueError("非法角色")

# 二、MCP上下文封装器
class MCPContext:
    def __init__(self, session_id=None):
        self.session_id = session_id or str(uuid.uuid4())
        self.history: List[MCPMessage] = []

    def add(self, role: str, content: str, call_type="prompt"):
        self.history.append(MCPMessage(role, content, call_type))

    def get_langchain_messages(self):
        return [msg.to_langchain() for msg in self.history]

    def get_last_intent(self) -> str:
        for msg in reversed(self.history):
            if msg.role == "user" and "查询" in msg.content:
                return "qa"
            elif msg.role == "user" and "执行" in msg.content:
                return "tool"
        return "dialog"

# 三、上下文路由器
class MCPRouter:
    def __init__(self):
        self.qwen = Qwen2Chat(model="Qwen/Qwen1.5-7B-Chat")
        self.deepseek = DeepSeekChat(model="deepseek-ai/deepseek-llm-chat")

    def route(self, context: MCPContext) -> str:
        intent = context.get_last_intent()
        print(f"[路由] 当前识别意图类型：{intent}")

        if intent == "qa":
            model = self.qwen
        elif intent == "tool":
            model = self.deepseek
        else:
            model = self.qwen  # 默认使用Qwen处理闲聊等

        try:
            messages = context.get_langchain_messages()
            response = model(messages)
            return response.content.strip()
        except Exception as e:
            return f"[ERROR] 模型调用失败：{str(e)}"

# 四、测试模拟运行
def run_demo():
    context = MCPContext()
    context.add("system", "你是一个知识丰富的智能助手")
    context.add("user", "请帮我查询一下2023年中国GDP是多少")
    context.add("assistant", "好的，我正在查询相关数据...")
    context.add("user", "此外，还请帮我执行一个统计脚本")

    router = MCPRouter()
    result = router.route(context)
    print("[模型响应] ", result)

run_demo()




# 【例12-7】
# -*- coding:utf-8 -*-
# A2A消息协议实现：Agent间模块消息传递与注入

import uuid
from typing import Dict, Callable, Any
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain.schema import HumanMessage, AIMessage

# 一、A2A标准消息结构
class A2AMessage:
    def __init__(self, sender: str, receiver: str, payload: str, metadata: Dict[str, Any] = None):
        self.message_id = str(uuid.uuid4())
        self.sender = sender
        self.receiver = receiver
        self.payload = payload
        self.metadata = metadata or {}

    def to_dict(self):
        return {
            "id": self.message_id,
            "sender": self.sender,
            "receiver": self.receiver,
            "payload": self.payload,
            "metadata": self.metadata,
        }

# 二、定义Agent基类
class Agent:
    def __init__(self, name: str, model_type="qwen"):
        self.name = name
        self.handlers: Dict[str, Callable[[A2AMessage], str]] = {}
        self.model = self._load_model(model_type)

    def _load_model(self, model_type):
        if model_type == "qwen":
            return Qwen2Chat(model="Qwen/Qwen1.5-7B-Chat")
        elif model_type == "deepseek":
            return DeepSeekChat(model="deepseek-ai/deepseek-llm-chat")
        else:
            raise ValueError("未知模型类型")

    def register_handler(self, task_type: str, handler: Callable[[A2AMessage], str]):
        self.handlers[task_type] = handler

    def receive_message(self, message: A2AMessage):
        task = message.metadata.get("task_type", "default")
        handler = self.handlers.get(task)
        if not handler:
            return f"[{self.name}] 不支持的任务类型：{task}"
        return handler(message)

    def default_handler(self, message: A2AMessage):
        prompt = f"请根据以下内容生成回应：{message.payload}"
        return self.model([HumanMessage(content=prompt)]).content.strip()

# 三、示例Agent：ToolAgent、AnswerAgent
class ToolAgent(Agent):
    def __init__(self):
        super().__init__("ToolAgent", model_type="deepseek")
        self.register_handler("tool_call", self.handle_tool)

    def handle_tool(self, msg: A2AMessage):
        tool_name = msg.metadata.get("tool", "未知工具")
        return f"[{self.name}] 已调用工具：{tool_name}，执行内容为：{msg.payload}"

class AnswerAgent(Agent):
    def __init__(self):
        super().__init__("AnswerAgent", model_type="qwen")
        self.register_handler("qa", self.handle_qa)

    def handle_qa(self, msg: A2AMessage):
        prompt = f"请回答问题：{msg.payload}"
        return self.model([HumanMessage(content=prompt)]).content.strip()

# 四、A2A调度中心
class A2ARouter:
    def __init__(self):
        self.agents = {}

    def register_agent(self, agent: Agent):
        self.agents[agent.name] = agent

    def dispatch(self, msg: A2AMessage) -> str:
        receiver = msg.receiver
        if receiver not in self.agents:
            return f"[ERROR] 目标Agent不存在：{receiver}"
        return self.agents[receiver].receive_message(msg)

# 五、运行示例
def run_demo():
    tool_agent = ToolAgent()
    qa_agent = AnswerAgent()

    router = A2ARouter()
    router.register_agent(tool_agent)
    router.register_agent(qa_agent)

    msg1 = A2AMessage(
        sender="MainAgent",
        receiver="ToolAgent",
        payload="请将CSV文件中的数据进行分析",
        metadata={"task_type": "tool_call", "tool": "DataAnalyzer"}
    )

    msg2 = A2AMessage(
        sender="MainAgent",
        receiver="AnswerAgent",
        payload="什么是Transformer架构？",
        metadata={"task_type": "qa"}
    )

    print("消息1响应：", router.dispatch(msg1))
    print("消息2响应：", router.dispatch(msg2))

run_demo()




# 【例12-8】
# -*- coding:utf-8 -*-
# LangChain主控Agent集成（Qwen3.0 + Deepseek + 多工具 + 记忆链）

import traceback
from typing import Optional
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.agents.agent import AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain_core.tools import tool

# 一、定义工具函数（可扩展多个）
@tool
def get_temperature(city: str) -> str:
    """根据城市名称获取当前温度（模拟）"""
    if city.lower() == "月球":
        raise ValueError("未找到对应城市")
    return f"{city}当前温度为26°C"

@tool
def multiply(a: str, b: str) -> str:
    """将两个数字相乘"""
    try:
        return str(float(a) * float(b))
    except:
        return "输入格式不正确，请输入数字"

TOOLS = [
    Tool.from_function(get_temperature),
    Tool.from_function(multiply),
]

# 二、定义支持LangChain的模型加载器
class ModelWithFallback:
    def __init__(self):
        self.qwen = Qwen2Chat(model="Qwen/Qwen1.5-7B-Chat")
        self.deepseek = DeepSeekChat(model="deepseek-ai/deepseek-llm-chat")

    def invoke(self, messages):
        try:
            return self.qwen(messages)
        except Exception as e:
            print("[WARN] Qwen调用失败，尝试Deepseek")
            try:
                return self.deepseek(messages)
            except Exception:
                print("[ERROR] 所有模型调用失败")
                return "系统暂时无法响应"

# 三、构建主控Agent逻辑
class LangChainAgent:
    def __init__(self):
        self.model = ModelWithFallback()
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        self.executor: Optional[AgentExecutor] = None
        self._init_agent()

    def _init_agent(self):
        # 使用LangChain内置Agent初始化方法
        self.executor = initialize_agent(
            tools=TOOLS,
            llm=self.model.qwen,  # 注意这里只能填具体实例，切换在外部处理
            agent=AgentType.OPENAI_FUNCTIONS,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True
        )

    def run(self, input_text: str) -> str:
        try:
            result = self.executor.run(input_text)
            return result
        except Exception as e:
            print("[ERROR] 执行出错:", traceback.format_exc())
            return "调用失败，请稍后重试"

# 四、模拟交互流程
def run_demo():
    agent = LangChainAgent()
    inputs = [
        "请帮我查询北京的当前温度",
        "将7和8相乘是多少",
        "请查一下月球的天气",
        "再帮我乘一下12和13",
    ]

    for i, text in enumerate(inputs):
        print(f"\n输入{i+1}：{text}")
        reply = agent.run(text)
        print("输出结果：", reply)

run_demo()




# 【例12-9】
# -*- coding:utf-8 -*-
# 工具调用正确率测试系统（Qwen + Deepseek + LangChain Tools）

from typing import List
from langchain.agents import initialize_agent, Tool, AgentExecutor, AgentType
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain_core.tools import tool

# 一、定义测试工具函数
@tool
def weather_tool(city: str) -> str:
    """模拟天气查询工具"""
    if city.lower() not in ["北京", "上海", "广州"]:
        raise ValueError("城市不支持")
    return f"{city}天气为晴，温度25°C"

@tool
def multiply_tool(a: str, b: str) -> str:
    """模拟乘法计算工具"""
    try:
        return str(float(a) * float(b))
    except:
        raise ValueError("输入格式错误")

TOOLS = [
    Tool.from_function(weather_tool),
    Tool.from_function(multiply_tool),
]

# 二、构建工具测试Agent
class ToolTestAgent:
    def __init__(self, model_type="qwen"):
        self.model_type = model_type
        self.llm = self._load_model()
        self.executor = initialize_agent(
            tools=TOOLS,
            llm=self.llm,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=False,
            handle_parsing_errors=True
        )

    def _load_model(self):
        if self.model_type == "qwen":
            return Qwen2Chat(model="Qwen/Qwen1.5-7B-Chat")
        elif self.model_type == "deepseek":
            return DeepSeekChat(model="deepseek-ai/deepseek-llm-chat")
        else:
            raise ValueError("不支持的模型类型")

    def run(self, prompt: str) -> str:
        try:
            return self.executor.run(prompt)
        except Exception as e:
            return f"[ERROR] {str(e)}"

# 三、定义测试集与执行框架
TEST_CASES = [
    {"input": "请查询一下北京的天气", "expected_tool": "weather_tool"},
    {"input": "上海今天气温多少？", "expected_tool": "weather_tool"},
    {"input": "广州天气怎么样？", "expected_tool": "weather_tool"},
    {"input": "请帮我计算7乘以8", "expected_tool": "multiply_tool"},
    {"input": "12和13相乘是多少？", "expected_tool": "multiply_tool"},
    {"input": "你知道北京和月球的温度吗？", "expected_tool": "weather_tool"},  # 故意触发错误
]

def run_tool_test(agent_type="qwen"):
    agent = ToolTestAgent(model_type=agent_type)
    correct = 0
    total = len(TEST_CASES)
    print(f"\n【开始测试】模型：{agent_type.upper()} | 总用例数：{total}")

    for i, case in enumerate(TEST_CASES):
        print(f"\n测试{i+1} 输入：{case['input']}")
        output = agent.run(case["input"])
        print("输出结果：", output)

        if case["expected_tool"] in output:
            print("[✓] 匹配成功")
            correct += 1
        else:
            print("[✗] 匹配失败")

    print(f"\n【测试完成】模型：{agent_type.upper()} 正确率：{correct}/{total} = {correct/total:.2f}")

# 四、运行主测试
def run_demo():
    run_tool_test("qwen")
    run_tool_test("deepseek")

run_demo()




# 【例12-10】
# -*- coding:utf-8 -*-
# 多用户并发测试与系统压测脚本（Qwen3.0 + Deepseek + LangChain）

import time
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain.agents import initialize_agent, Tool, AgentType
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain_core.tools import tool

# 一、定义基础工具函数
@tool
def fake_weather(city: str) -> str:
    return f"{city}天气晴朗，26°C"

@tool
def calc(a: str, b: str) -> str:
    try:
        return str(float(a) * float(b))
    except:
        return "输入有误"

TOOLS = [
    Tool.from_function(fake_weather),
    Tool.from_function(calc),
]

# 二、定义模型Agent执行器（支持Qwen/Deepseek）
class ConcurrentAgent:
    def __init__(self, model_type="qwen"):
        self.model_type = model_type
        self.model = self._load_model()
        self.executor = initialize_agent(
            tools=TOOLS,
            llm=self.model,
            agent=AgentType.OPENAI_FUNCTIONS,
            verbose=False,
            handle_parsing_errors=True,
        )

    def _load_model(self):
        if self.model_type == "qwen":
            return Qwen2Chat(model="Qwen/Qwen1.5-7B-Chat")
        elif self.model_type == "deepseek":
            return DeepSeekChat(model="deepseek-ai/deepseek-llm-chat")
        else:
            raise ValueError("未知模型类型")

    def run(self, text: str) -> str:
        try:
            return self.executor.run(text)
        except Exception as e:
            return f"[ERROR] {str(e)}"

# 三、并发测试器
def run_concurrent_test(concurrent_users: int = 20, model_type="qwen"):
    agent = ConcurrentAgent(model_type=model_type)
    tasks = [
        "请查询北京的天气",
        "广州今天什么天气",
        "请帮我计算 12 * 13",
        "算一下9乘以7是多少",
        "帮我查查上海的气温",
    ]

    results = []
    start_time = time.time()

    def task_runner(user_id: int):
        text = random.choice(tasks)
        st = time.time()
        response = agent.run(text)
        et = time.time()
        duration = et - st
        return {
            "user_id": user_id,
            "input": text,
            "output": response,
            "duration": duration
        }

    print(f"\n【并发测试启动】模型：{model_type.upper()} | 用户数：{concurrent_users}")
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = [executor.submit(task_runner, i) for i in range(concurrent_users)]
        for f in as_completed(futures):
            results.append(f.result())

    end_time = time.time()
    total_time = end_time - start_time

    # 输出统计指标
    success_count = sum(1 for r in results if "[ERROR]" not in r["output"])
    avg_time = sum(r["duration"] for r in results) / len(results)
    qps = len(results) / total_time

    print("\n【测试结果统计】")
    print(f"总请求数：{len(results)}")
    print(f"成功响应数：{success_count}")
    print(f"失败响应数：{len(results) - success_count}")
    print(f"平均响应时间：{avg_time:.2f}s")
    print(f"QPS（每秒处理请求数）：{qps:.2f}")

    # 输出部分响应详情
    for r in results[:5]:
        print(f"\n用户{r['user_id']}输入：{r['input']}")
        print(f"响应用时：{r['duration']:.2f}s")
        print(f"响应内容：{r['output']}")

# 四、测试入口
def run_demo():
    run_concurrent_test(concurrent_users=20, model_type="qwen")
    run_concurrent_test(concurrent_users=20, model_type="deepseek")

run_demo()




# 【例12-11】
# -*- coding:utf-8 -*-
# 模型幻觉率与满意度评估系统（Qwen + Deepseek）

import random
import difflib
from langchain_community.chat_models import Qwen2Chat, DeepSeekChat
from langchain.schema import HumanMessage

# 一、定义基础问答数据集（真实QA + 标准答案）
EVAL_SET = [
    {
        "question": "请问2022年中国的GDP是多少？",
        "reference": "2022年中国GDP约为121万亿元人民币",
    },
    {
        "question": "地球上最大的哺乳动物是什么？",
        "reference": "地球上最大的哺乳动物是蓝鲸",
    },
    {
        "question": "爱因斯坦是哪一年获得诺贝尔奖的？",
        "reference": "爱因斯坦于1921年获得诺贝尔物理学奖",
    },
    {
        "question": "Transformer模型由哪位研究者提出？",
        "reference": "Transformer由Google的Vaswani等人在2017年提出",
    },
    {
        "question": "请列出中国的四大发明",
        "reference": "中国四大发明包括造纸术、指南针、火药和印刷术",
    }
]

# 二、加载模型接口
class EvalAgent:
    def __init__(self, model_type="qwen"):
        if model_type == "qwen":
            self.model = Qwen2Chat(model="Qwen/Qwen1.5-7B-Chat")
        elif model_type == "deepseek":
            self.model = DeepSeekChat(model="deepseek-ai/deepseek-llm-chat")
        else:
            raise ValueError("不支持的模型类型")

    def ask(self, text: str) -> str:
        try:
            msg = [HumanMessage(content=text)]
            response = self.model(msg)
            return response.content.strip()
        except Exception as e:
            return f"[ERROR] 模型调用失败：{str(e)}"

# 三、幻觉检测算法（基于规则与人工比对）
def hallucination_detect(generated: str, reference: str) -> bool:
    seq = difflib.SequenceMatcher(None, generated, reference)
    similarity = seq.ratio()
    return similarity < 0.6  # 小于60%视为严重偏离

# 四、模拟用户满意度评分（随机+权重）
def user_score(generated: str) -> int:
    if "[ERROR]" in generated:
        return 1
    base = random.randint(3, 5)
    if "不知道" in generated or "无法回答" in generated:
        return base - 2
    return base

# 五、评估主流程
def run_evaluation(model_type="qwen"):
    agent = EvalAgent(model_type=model_type)
    hallucinated = 0
    total = len(EVAL_SET)
    total_score = 0

    print(f"\n【评估模型：{model_type.upper()}】")

    for i, item in enumerate(EVAL_SET):
        print(f"\n问题{i+1}：{item['question']}")
        answer = agent.ask(item["question"])
        print("生成回答：", answer)
        print("参考答案：", item["reference"])
        is_hallucinated = hallucination_detect(answer, item["reference"])
        score = user_score(answer)
        total_score += score
        if is_hallucinated:
            print("[×] 存在幻觉")
            hallucinated += 1
        else:
            print("[✓] 无幻觉")
        print(f"用户评分（模拟）：{score}")

    halluc_rate = hallucinated / total
    avg_score = total_score / total

    print("\n【评估结果统计】")
    print(f"总问题数：{total}")
    print(f"幻觉数量：{hallucinated}")
    print(f"幻觉率：{halluc_rate:.2f}")
    print(f"平均满意度得分（1~5）：{avg_score:.2f}")

# 六、执行评估
def run_demo():
    run_evaluation("qwen")
    run_evaluation("deepseek")

run_demo()
