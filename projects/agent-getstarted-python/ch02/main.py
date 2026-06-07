# 第2章 智能体系统的组成结构与运行机制
# 【例2-1】
# agent_startup.py

import logging
import datetime
from typing import List
from qwen_agent import Agent
from qwen_agent.tools.base import BaseTool

Tool = BaseTool
ChatMessage = dict


class BaseMemory:
    pass


class SimpleMemory(BaseMemory):
    def __init__(self):
        self.messages = []

    def add_message(self, message: dict):
        self.messages.append(message)

    def get(self):
        return self.messages


def llm_config(model: str = "qwen-plus", **generate_cfg):
    return {
        "model": model,
        "generate_cfg": generate_cfg,
    }

# 配置日志
logging.basicConfig(
    filename='agent_startup.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# 初始化工具：当前时间工具
class GetTimeTool(BaseTool):
    def run(self, params: dict) -> str:
        now = datetime.datetime.now()
        return now.strftime("当前时间是：%Y年%m月%d日 %H:%M:%S")

    @property
    def description(self):
        return "获取当前系统时间"

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

# 初始化工具：任务初始化检查工具
class StartupCheckTool(BaseTool):
    def run(self, params: dict) -> str:
        checks = ["模型加载完成", "工具已注册", "Memory注入成功", "上下文初始化完成"]
        return "系统初始化检查通过：" + "，".join(checks)

    @property
    def description(self):
        return "执行Agent启动时的系统检查任务"

    @property
    def parameters(self):
        return {"type": "object", "properties": {}}

# 构建Agent对象
def build_agent() -> Agent:
    logging.info("启动Agent构建流程")
    
    llm = llm_config(
        temperature=0.3,
        max_tokens=512
    )

    tools: List[Tool] = [GetTimeTool(), StartupCheckTool()]

    agent = Agent(
        name="SystemStartupAgent",
        llm=llm,
        tools=tools,
        system_message="你是一位系统智能体助手，负责初始化流程、工具检查与运行日志监控。"
    )

    logging.info("Agent构建完成")
    return agent

# 启动测试交互
def run_startup_sequence():
    agent = build_agent()

    logging.info("执行工具调用测试")
    res1 = agent.chat("请执行一次系统检查")
    print(">> 系统检查响应：", res1)

    logging.info("执行时间工具测试")
    res2 = agent.chat("请告诉我现在几点")
    print(">> 当前时间响应：", res2)

    logging.info("Agent初始化流程全部完成")

if __name__ == "__main__":
    run_startup_sequence()




# 【例2-2】
# agent_state_tracking.py

from qwen_agent import Agent
from qwen_agent.tools.base import BaseTool

# 工具一：记录会议主题
class SetTopicTool(BaseTool):
    def run(self, params: dict) -> str:
        topic = params.get("topic", "")
        return f"会议主题已设置为：{topic}"

    @property
    def description(self):
        return "设置会议主题"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "topic": {"type": "string", "description": "会议主题"}
            },
            "required": ["topic"]
        }

# 工具二：设置会议时间
class SetTimeTool(BaseTool):
    def run(self, params: dict) -> str:
        time = params.get("time", "")
        return f"会议时间已设置为：{time}"

    @property
    def description(self):
        return "设置会议时间"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "time": {"type": "string", "description": "会议时间"}
            },
            "required": ["time"]
        }

# 工具三：设置与会人
class SetAttendeesTool(BaseTool):
    def run(self, params: dict) -> str:
        attendees = params.get("attendees", "")
        return f"与会人员已设定为：{attendees}"

    @property
    def description(self):
        return "设置与会人员"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "attendees": {"type": "string", "description": "人员名单"}
            },
            "required": ["attendees"]
        }

# 构建Agent
def build_state_agent():
    llm = llm_config(
        temperature=0.3,
        max_tokens=512
    )

    memory = SimpleMemory()  # 基础记忆模块用于状态追踪

    tools = [SetTopicTool(), SetTimeTool(), SetAttendeesTool()]

    agent = Agent(
        name="MeetingPlannerAgent",
        llm=llm,
        tools=tools,
        memory=memory,
        system_message="你是一个会议助手Agent，请协助用户完成会议主题、时间和与会人员设定，需记录所有输入信息以备后续回顾"
    )

    return agent

# 运行模拟对话过程
def run_tracking_test():
    agent = build_state_agent()

    print(agent.chat("我要安排一个关于AI发展的会议"))
    print(agent.chat("会议时间定在5月3号上午10点"))
    print(agent.chat("参加人包括张三、李四和王五"))
    print(agent.chat("请总结一下目前的会议信息"))

if __name__ == "__main__":
    run_tracking_test()




# 【例2-3】
# persistent_memory_agent.py

import os
import json
from typing import List
from qwen_agent import Agent
from qwen_agent.tools.base import BaseTool


# 自定义Memory：将上下文持久化至JSON文件
class PersistentJSONMemory(BaseMemory):
    def __init__(self, session_id: str, path: str = "./memory"):
        os.makedirs(path, exist_ok=True)
        self.session_id = session_id
        self.memory_file = os.path.join(path, f"{session_id}.json")
        self.messages = self.load()

    def load(self) -> List[dict]:
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def save(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.messages, f, ensure_ascii=False, indent=2)

    def add_message(self, message: dict):
        self.messages.append(message)
        self.save()

    def get(self) -> List[dict]:
        return self.messages


# 示例工具：记录交互内容
class LogUserInputTool(BaseTool):
    def run(self, params: dict) -> str:
        content = params.get("content", "")
        return f"记录完成：{content}"

    @property
    def description(self):
        return "将用户输入写入交互日志"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "用户内容"}
            },
            "required": ["content"]
        }


# 构建Agent
def build_agent(session_id: str):
    llm = llm_config(
        temperature=0.3,
        max_tokens=512
    )

    memory = PersistentJSONMemory(session_id=session_id)

    tools = [LogUserInputTool()]

    agent = Agent(
        name="SessionRecoveryAgent",
        llm=llm,
        tools=tools,
        memory=memory,
        system_message="你是一位具备上下文持久化能力的智能体，可以记录并恢复中断的多轮交互内容"
    )

    return agent


# 模拟用户对话过程（中断→恢复）
def simulate_session():
    session_id = "user_20250501"

    agent = build_agent(session_id)

    print(agent.chat("我想订一张5月5日去上海的机票"))
    print(agent.chat("时间大约是早上8点左右"))
    print(agent.chat("帮我记录这个需求"))

    print("\n模拟中断后重新启动...\n")

    agent2 = build_agent(session_id)

    print(agent2.chat("现在继续，出发地是北京"))

if __name__ == "__main__":
    simulate_session()




# 【例2-4】
# agent_shutdown_cleanup.py

import os
import json
import datetime
from typing import List
from qwen_agent import Agent
from qwen_agent.tools.base import BaseTool

# Agent使用的本地存储路径
MEMORY_PATH = "./memory_cleanup"

# 自定义Memory类，支持写入日志文件
class LoggingMemory(BaseMemory):
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory_file = os.path.join(MEMORY_PATH, f"{session_id}.json")
        self.logs = []

    def add_message(self, message: dict):
        self.logs.append(message)

    def get(self) -> List[dict]:
        return self.logs

    def save(self):
        os.makedirs(MEMORY_PATH, exist_ok=True)
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.logs, f, ensure_ascii=False, indent=2)

    def clear(self):
        if os.path.exists(self.memory_file):
            os.remove(self.memory_file)

# 工具：模拟任务处理
class TaskTool(BaseTool):
    def run(self, params: dict) -> str:
        task = params.get("task", "")
        return f"任务『{task}』已完成"

    @property
    def description(self):
        return "处理一个示例任务"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "任务内容"}
            },
            "required": ["task"]
        }

# Agent对象构建
class ManagedAgent:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.memory = LoggingMemory(session_id)
        self.agent = Agent(
            name="CleanableAgent",
            llm=llm_config(
                temperature=0.2,
                max_tokens=256
            ),
            tools=[TaskTool()],
            memory=self.memory,
            system_message="你是一位临时任务助手，任务完成后需要注销并清理所有资源"
        )

    def chat(self, user_input: str) -> str:
        result = self.agent.chat(user_input)
        return result

    def shutdown(self):
        print(">> 执行Agent注销流程...")
        self.memory.save()
        self.memory.clear()
        log_file = os.path.join(MEMORY_PATH, f"{self.session_id}_log.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            for item in self.memory.logs:
                ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"[{ts}] {item}\n")
        print(">> Agent已成功注销并释放资源")

# 模拟流程：运行→完成→注销
def run_task_session():
    session_id = "temp_agent_001"
    agent = ManagedAgent(session_id)

    print(agent.chat("请帮我完成今天的日报任务"))
    print(agent.chat("现在关闭智能体"))

    agent.shutdown()

if __name__ == "__main__":
    run_task_session()





# 【例2-5】
# agent_web_api_weather.py

import requests
import json
from qwen_agent import Agent
from qwen_agent.tools.base import BaseTool

# 工具：天气查询工具
class WeatherAPITool(BaseTool):
    def run(self, params: dict) -> str:
        city = params.get("city", "").strip()
        if not city:
            return "请输入有效的城市名"
        try:
            # 调用 wttr.in 接口
            url = f"https://wttr.in/{city}?format=j1"
            resp = requests.get(url, timeout=5)
            if resp.status_code != 200:
                return f"查询失败，状态码：{resp.status_code}"
            data = resp.json()
            current = data["current_condition"][0]
            temp = current["temp_C"]
            humidity = current["humidity"]
            desc = current["weatherDesc"][0]["value"]
            return f"{city}当前天气：{desc}，温度：{temp}℃，湿度：{humidity}%"
        except Exception as e:
            return f"调用天气API出错：{str(e)}"

    @property
    def description(self):
        return "获取指定城市的实时天气信息"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"}
            },
            "required": ["city"]
        }

# 构建Agent对象
def build_weather_agent():
    agent = Agent(
        name="WeatherQueryAgent",
        llm=llm_config(
            temperature=0.3,
            max_tokens=256
        ),
        tools=[WeatherAPITool()],
        system_message="你是一位天气查询助手，支持调用外部API获取城市实时天气信息"
    )
    return agent

# 模拟用户查询天气
def run_weather_query():
    agent = build_weather_agent()

    print(agent.chat("请告诉我北京现在的天气"))
    print(agent.chat("上海的天气怎么样？"))
    print(agent.chat("请查询一下东京的当前温度和湿度"))

if __name__ == "__main__":
    run_weather_query()





# 【例2-6】
# agent_sqlite_db.py

import sqlite3
from qwen_agent import Agent
from qwen_agent.tools.base import BaseTool

DB_PATH = "./employee.db"

# 初始化数据库（首次创建）
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            department TEXT,
            title TEXT
        )
    ''')
    conn.commit()
    conn.close()

# 工具：添加员工
class AddEmployeeTool(BaseTool):
    def run(self, params: dict) -> str:
        name = params.get("name")
        department = params.get("department", "")
        title = params.get("title", "")
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO employees (name, department, title) VALUES (?, ?, ?)",
                (name, department, title)
            )
            conn.commit()
            return f"员工{name}添加成功"
        except Exception as e:
            return f"添加失败：{str(e)}"
        finally:
            conn.close()

    @property
    def description(self):
        return "添加一名员工信息"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "员工姓名"},
                "department": {"type": "string", "description": "所属部门"},
                "title": {"type": "string", "description": "职务"}
            },
            "required": ["name"]
        }

# 工具：查询员工
class QueryEmployeeTool(BaseTool):
    def run(self, params: dict) -> str:
        name = params.get("name", "")
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            if name:
                cursor.execute("SELECT name, department, title FROM employees WHERE name = ?", (name,))
            else:
                cursor.execute("SELECT name, department, title FROM employees")
            rows = cursor.fetchall()
            if not rows:
                return "未找到匹配员工"
            return "\n".join([f"{n} | {d} | {t}" for n, d, t in rows])
        except Exception as e:
            return f"查询失败：{str(e)}"
        finally:
            conn.close()

    @property
    def description(self):
        return "查询员工信息"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "员工姓名（可选）"}
            }
        }

# 构建Agent
def build_db_agent():
    agent = Agent(
        name="EmployeeDBAgent",
        llm=llm_config(temperature=0.2),
        tools=[AddEmployeeTool(), QueryEmployeeTool()],
        system_message="你是一位人事数据库助手，能添加和查询员工信息，信息包含姓名、部门与职务"
    )
    return agent

# 模拟用户对话流程
def run_demo():
    init_db()
    agent = build_db_agent()

    print(agent.chat("请添加一名员工，姓名是王小明，部门是技术部，职位是高级工程师"))
    print(agent.chat("请查询王小明的详细信息"))
    print(agent.chat("我想看看所有员工列表"))

if __name__ == "__main__":
    run_demo()




# 【例2-7】
# agent_file_exec.py

import os
import traceback
from qwen_agent import Agent
from qwen_agent.tools.base import BaseTool

# 工具：读取文件内容
class ReadFileTool(BaseTool):
    def run(self, params: dict) -> str:
        path = params.get("filepath", "")
        if not os.path.exists(path):
            return f"文件不存在：{path}"
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            return f"读取失败：{str(e)}"

    @property
    def description(self):
        return "读取指定文件内容"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "文件路径"}
            },
            "required": ["filepath"]
        }

# 工具：写入文件内容
class WriteFileTool(BaseTool):
    def run(self, params: dict) -> str:
        path = params.get("filepath", "")
        content = params.get("content", "")
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return f"内容已成功写入文件：{path}"
        except Exception as e:
            return f"写入失败：{str(e)}"

    @property
    def description(self):
        return "将文本写入指定文件"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "文件路径"},
                "content": {"type": "string", "description": "写入内容"}
            },
            "required": ["filepath", "content"]
        }

# 工具：执行Python代码
class ExecCodeTool(BaseTool):
    def run(self, params: dict) -> str:
        code = params.get("code", "")
        try:
            local_vars = {}
            exec(code, {}, local_vars)
            return "执行成功，输出变量：" + str(local_vars)
        except Exception:
            return "执行出错：" + traceback.format_exc(limit=2)

    @property
    def description(self):
        return "执行用户提供的Python代码（仅限受控环境）"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "待执行Python代码"}
            },
            "required": ["code"]
        }

# 构建Agent
def build_agent():
    agent = Agent(
        name="FileExecAgent",
        llm=llm_config(temperature=0.2),
        tools=[ReadFileTool(), WriteFileTool(), ExecCodeTool()],
        system_message="你是一位具备文件系统与代码执行能力的助手，负责文本文件处理与Python代码运行"
    )
    return agent

# 测试流程
def run_demo():
    agent = build_agent()

    print(agent.chat("请将以下内容写入文件test.txt：你好，这是智能体写入的内容"))
    print(agent.chat("读取文件test.txt中的内容"))
    print(agent.chat("请运行如下代码：a = 5\nb = 3\nc = a * b"))

if __name__ == "__main__":
    run_demo()




# 【例2-8】
# agent_ui_api.py

from fastapi import FastAPI
from pydantic import BaseModel
from qwen_agent import Agent
from fastapi.middleware.cors import CORSMiddleware

## UI API 请求结构
class ChatRequest(BaseModel):
    session_id: str
    message: str

class ChatResponse(BaseModel):
    session_id: str
    reply: str

## Agent 定义（模拟记账机器人）
class AddRecordTool(Tool):
    def run(self, params: dict) -> str:
        category = params.get("category")
        amount = params.get("amount")
        return f"已记录消费：{category} - {amount}元"

    @property
    def description(self):
        return "记录一笔消费信息"

    @property
    def parameters(self):
        return {
            "type": "object",
            "properties": {
                "category": {"type": "string", "description": "消费类别"},
                "amount": {"type": "string", "description": "金额（单位：元）"}
            },
            "required": ["category", "amount"]
        }

def build_agent() -> Agent:
    return Agent(
        name="FinanceBot",
        llm=llm_config(temperature=0.3),
        tools=[AddRecordTool()],
        system_message="你是一个家庭记账助手，接收用户输入并记录消费类型与金额"
    )

## FastAPI 服务构建
app = FastAPI()
agent_instance = build_agent()

# CORS 允许跨域（用于网页调用）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# 根接口
@app.get("/")
def index():
    return {"message": "Agent API 运行中"}

# 聊天API：接收用户输入→Agent处理→返回输出
@app.post("/chat", response_model=ChatResponse)
def chat_with_agent(req: ChatRequest):
    reply = agent_instance.chat(req.message)
    return ChatResponse(session_id=req.session_id, reply=reply)






# 使用如下命令启动脚本：
# uvicorn agent_ui_api:app --reload --port 8080
# 使用如下方式请求API：
# curl -X POST http://localhost:8080/chat \
#   -H "Content-Type: application/json" \
#   -d '{"session_id": "user123", "message": "刚刚买菜花了20元"}'
