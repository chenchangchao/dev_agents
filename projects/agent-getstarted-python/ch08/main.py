# 第8章 多智能体系统构建实战
# 【例8-1】
import time
from typing import Dict, List
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

# ========= 资讯Agent定义 =========
class NewsAgent(Tool):
    def __init__(self):
        super().__init__(name="news_agent", description="抓取并提取财经新闻")

    def call(self, query: str) -> str:
        # 模拟新闻摘要
        return f"[资讯摘要] 当前宏观新闻显示：制造业回暖，CPI保持温和，出口增长稳定。"

# ========= 分析Agent定义 =========
class ForecastAgent(Tool):
    def __init__(self):
        super().__init__(name="forecast_agent", description="根据资讯预测GDP趋势")

    def call(self, summary: str) -> str:
        return f"[预测结果] 综合判断2024年GDP预计增长5.3%。分析依据为：{summary}"

# ========= 主控Agent封装 =========
class ControlAgent:
    def __init__(self, tools: List[Tool]):
        self.agent = Agent(tools=tools)
        self.news_tool = tools[0]
        self.forecast_tool = tools[1]

    def run(self, user_query: str) -> str:
        print("→ 开始执行主控Agent任务拆解与调度")
        
        # Step 1：交由新闻Agent处理
        print("→ 调用新闻Agent进行资讯提取...")
        summary = self.news_tool.call(user_query)
        print(f"→ 资讯Agent返回：{summary}")

        # Step 2：交由分析Agent进行预测
        print("→ 调用预测Agent进行经济建模...")
        forecast = self.forecast_tool.call(summary)
        print(f"→ 分析Agent返回：{forecast}")

        # Step 3：整理回应交由Qwen输出
        final_prompt = f"""请基于以下任务链条返回统一回应：

任务目标：{user_query}
资讯摘要：{summary}
预测输出：{forecast}
请以财经顾问语气生成最终报告："""

        result = self.agent.chat(messages=[Message(role="user", content=final_prompt)])
        return result.content

# ========= 主流程运行 =========
if __name__ == "__main__":
    print("=== 启动基于职责建模的多Agent系统 ===")

    news = NewsAgent()
    forecast = ForecastAgent()
    control_agent = ControlAgent(tools=[news, forecast])

    query = "请预测2024年中国GDP走势，并说明依据"
    response = control_agent.run(query)

    print("\n--- Qwen3.0最终响应 ---")
    print(response)




# 【例8-2】
import time
from typing import Dict, Any
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

# ========= 全局状态管理器 =========
class SharedState:
    def __init__(self):
        self.state: Dict[str, Any] = {}

    def update(self, key: str, value: Any):
        print(f"→ [状态更新] {key} = {value}")
        self.state[key] = value

    def get(self, key: str, default=None):
        return self.state.get(key, default)

    def dump(self) -> Dict[str, Any]:
        return self.state.copy()

# ========= 分析Agent =========
class AnalysisAgent(Tool):
    def __init__(self, shared_state: SharedState):
        super().__init__(name="analysis_agent", description="执行数据分析任务")
        self.shared = shared_state

    def call(self, query: str) -> str:
        # 假设输入为“2024年宏观经济趋势”
        result = "制造业增长、CPI温和回落、出口回升"
        self.shared.update("econ_trend", result)
        return f"[分析完成] 趋势：{result}"

# ========= 报告Agent =========
class ReportAgent(Tool):
    def __init__(self, shared_state: SharedState):
        super().__init__(name="report_agent", description="根据分析内容撰写最终报告")
        self.shared = shared_state

    def call(self, _: str) -> str:
        trend = self.shared.get("econ_trend", "无数据")
        report = f"2024年经济展望：预计{trend}，将推动GDP稳定增长。"
        self.shared.update("final_report", report)
        return report

# ========= 主控Agent =========
class ControllerAgent:
    def __init__(self, tools: list, shared_state: SharedState):
        self.agent = Agent(tools=tools)
        self.shared = shared_state
        self.analysis_tool = tools[0]
        self.report_tool = tools[1]

    def execute(self, user_query: str) -> str:
        print("\n→ 主控Agent启动，任务分析中...")
        self.shared.update("user_input", user_query)

        # Step 1: 调用分析Agent
        analysis = self.analysis_tool.call(user_query)
        print("→ 分析Agent响应：", analysis)

        # Step 2: 调用报告Agent
        report = self.report_tool.call("")
        print("→ 报告Agent响应：", report)

        # Step 3: 最终合成响应
        prompt = f"""请基于以下信息生成一段完整用户报告：
用户问题：{self.shared.get('user_input')}
分析内容：{self.shared.get('econ_trend')}
最终报告：{self.shared.get('final_report')}"""

        result = self.agent.chat(messages=[Message(role="user", content=prompt)])
        return result.content

# ========= 主流程 =========
if __name__ == "__main__":
    print("=== 启动多Agent状态共享系统 ===")
    shared = SharedState()
    agent1 = AnalysisAgent(shared)
    agent2 = ReportAgent(shared)
    controller = ControllerAgent([agent1, agent2], shared)

    query = "请分析2024年中国经济走势，并撰写总结报告"
    final_response = controller.execute(query)

    print("\n--- Qwen3.0最终响应 ---")
    print(final_response)




# 【例8-3】
import random
from typing import List, Dict
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

# ========= 模拟多个财经Agent =========
class EconomicAgent(Tool):
    def __init__(self, name, quality: float):
        desc = f"财经预测Agent（性能评分：{quality}）"
        super().__init__(name=name, description=desc)
        self.quality = quality

    def call(self, query: str) -> str:
        return f"[{self.name}] 分析完成：预计增长为 {round(5 + random.random(), 2)}%。性能评分为 {self.quality}"

# ========= 调度器定义 =========
class TaskDispatcher:
    def __init__(self, agents: List[Tool], strategy: str = "round_robin", weights: Dict[str, int] = None):
        self.agents = agents
        self.strategy = strategy
        self.weights = weights or {}
        self.counter = 0
        self.pool = self._build_weighted_pool() if strategy == "weighted" else agents

    def _build_weighted_pool(self):
        pool = []
        for agent in self.agents:
            weight = self.weights.get(agent.name, 1)
            pool.extend([agent] * weight)
        return pool

    def dispatch(self) -> Tool:
        if self.strategy == "round_robin":
            agent = self.agents[self.counter % len(self.agents)]
            self.counter += 1
            return agent
        elif self.strategy == "weighted":
            return random.choice(self.pool)

# ========= 主控系统 =========
class ControlSystem:
    def __init__(self, dispatcher: TaskDispatcher):
        self.dispatcher = dispatcher
        self.agent = Agent(tools=dispatcher.agents)

    def run_task(self, query: str) -> str:
        tool = self.dispatcher.dispatch()
        print(f"\n→ 当前调度策略：{self.dispatcher.strategy.upper()}")
        print(f"→ 分配给Agent：{tool.name}")
        tool_output = tool.call(query)

        prompt = f"""请根据以下Agent的分析生成最终简明财经预测：

用户问题：{query}
Agent分析：{tool_output}
"""

        result = self.agent.chat(messages=[Message(role="user", content=prompt)])
        return result.content

# ========= 启动流程 =========
if __name__ == "__main__":
    print("=== 启动任务调度系统 ===")

    # 创建三个经济Agent副本
    agent1 = EconomicAgent("econ_agent_1", 0.85)
    agent2 = EconomicAgent("econ_agent_2", 0.90)
    agent3 = EconomicAgent("econ_agent_3", 0.95)

    # 轮询策略执行
    round_dispatcher = TaskDispatcher([agent1, agent2, agent3], strategy="round_robin")
    system_rr = ControlSystem(dispatcher=round_dispatcher)
    for _ in range(3):
        print(system_rr.run_task("预测2024年中国GDP增长"))

    # 加权策略执行
    weight_map = {"econ_agent_1": 1, "econ_agent_2": 1, "econ_agent_3": 3}
    weighted_dispatcher = TaskDispatcher([agent1, agent2, agent3], strategy="weighted", weights=weight_map)
    system_weighted = ControlSystem(dispatcher=weighted_dispatcher)
    for _ in range(3):
        print(system_weighted.run_task("预测2024年中国GDP增长"))




# 【例8-4】
import time
import heapq
from typing import Dict, List, Callable
from qwen_agent.agent import Agent
from qwen_agent.context import Message

# ========= 任务定义类 =========
class Task:
    def __init__(self, id: str, func: Callable, depends: List[str] = None, priority: int = 1):
        self.id = id
        self.func = func
        self.depends = depends or []
        self.priority = priority
        self.executed = False
        self.result = None

# ========= 任务图执行器 =========
class TaskExecutor:
    def __init__(self):
        self.tasks: Dict[str, Task] = {}
        self.execution_log = []

    def register(self, task: Task):
        self.tasks[task.id] = task

    def _can_execute(self, task: Task) -> bool:
        return all(self.tasks[dep].executed for dep in task.depends)

    def execute(self):
        pq = []
        for task in self.tasks.values():
            if not task.depends:
                heapq.heappush(pq, (-task.priority, task.id))

        while pq:
            _, tid = heapq.heappop(pq)
            task = self.tasks[tid]
            if task.executed:
                continue
            if not self._can_execute(task):
                continue

            print(f"\n→ 执行任务：{tid}（优先级：{task.priority}）")
            task.result = task.func()
            task.executed = True
            self.execution_log.append((tid, task.result))

            # 解锁下游任务
            for t in self.tasks.values():
                if not t.executed and self._can_execute(t):
                    heapq.heappush(pq, (-t.priority, t.id))

    def get_summary(self) -> str:
        return "\n".join([f"{tid}: {res}" for tid, res in self.execution_log])

# ========= 模拟Agent行为函数 =========
def task_fetch_news():
    time.sleep(1)
    return "抓取完成：2024年CPI增速放缓、出口回升"

def task_model_analysis():
    time.sleep(1)
    return "分析完成：预计GDP增长5.4%"

def task_generate_graph():
    time.sleep(1)
    return "图表渲染完成"

def task_generate_report():
    time.sleep(1)
    return "最终报告生成完毕"

# ========= 主控Agent整合调用 =========
if __name__ == "__main__":
    print("=== 启动带依赖与优先级调度的Agent任务系统 ===")
    executor = TaskExecutor()

    # 注册任务：ID、函数、依赖项、优先级（越大越高）
    executor.register(Task("fetch_news", task_fetch_news, priority=3))
    executor.register(Task("model", task_model_analysis, depends=["fetch_news"], priority=4))
    executor.register(Task("graph", task_generate_graph, depends=["model"], priority=1))
    executor.register(Task("report", task_generate_report, depends=["model"], priority=2))

    executor.execute()

    # 汇总并调用Qwen3.0生成总结
    agent = Agent()
    prompt = f"""以下是多任务智能体执行结果，请生成一段完整分析总结：

{executor.get_summary()}"""

    result = agent.chat(messages=[Message(role="user", content=prompt)])
    print("\n--- Qwen3.0输出总结 ---")
    print(result.content)




# 【例8-5】
import time
import threading
from typing import Dict
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

# ========= 子Agent定义 =========
class PolicyAgent(Tool):
    def __init__(self):
        super().__init__(name="policy_agent", description="分析最新宏观政策")

    def call(self, _: str) -> str:
        time.sleep(1.2)
        return "政策导向积极，财政刺激与减税政策持续发力。"

class SentimentAgent(Tool):
    def __init__(self):
        super().__init__(name="sentiment_agent", description="提取市场情绪")

    def call(self, _: str) -> str:
        time.sleep(1.5)
        return "市场情绪整体乐观，投资者信心显著回暖。"

class TrendAgent(Tool):
    def __init__(self):
        super().__init__(name="trend_agent", description="预测经济增长趋势")

    def call(self, _: str) -> str:
        time.sleep(1.7)
        return "预计2024年GDP增长为5.5%，主要来源于消费与出口复苏。"

# ========= 并行执行管理器 =========
class ParallelExecutor:
    def __init__(self):
        self.results: Dict[str, str] = {}
        self.lock = threading.Lock()

    def run_agent(self, agent: Tool, name: str):
        print(f"→ 启动{name}")
        result = agent.call("")
        with self.lock:
            self.results[name] = result
        print(f"→ {name}完成")

    def execute_all(self, agents: Dict[str, Tool]):
        threads = []
        for name, agent in agents.items():
            t = threading.Thread(target=self.run_agent, args=(agent, name))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        return self.results

# ========= 主控Agent系统 =========
if __name__ == "__main__":
    print("=== 启动子Agent并行执行系统 ===")

    # 注册子Agent
    agents = {
        "政策分析": PolicyAgent(),
        "市场情绪": SentimentAgent(),
        "趋势预测": TrendAgent()
    }

    # 并行执行
    executor = ParallelExecutor()
    results = executor.execute_all(agents)

    # 汇总生成Prompt
    prompt = "请根据以下信息撰写2024年一季度中国经济简报：\n"
    for title, content in results.items():
        prompt += f"\n【{title}】：{content}"

    print("\n→ 汇总Prompt：")
    print(prompt)

    # Qwen3.0生成最终响应
    agent = Agent()
    final_response = agent.chat(messages=[Message(role="user", content=prompt)])

    print("\n--- Qwen3.0响应输出 ---")
    print(final_response.content)






# 8.3.1 Agent间通信协议格式
# 以Qwen3.0智能体系统中应用的标准通信格式为例，一条完整的Agent通信消息如下所示（结构化抽象）：
{
  "id": "msg-20240503-00123",
  "timestamp": "2025-05-03T13:45:27Z",
  "from": "agent:task_dispatcher",
  "to": "agent:data_parser",
  "type": "command",
  "priority": 2,
  "context_ref": ["mem-1001", "task-req-879"],
  "payload": {
    "function": "parse_table",
    "args": {
      "file_url": "https://data.gov.cn/table.csv",
      "columns": ["region", "gdp"]
    }
  },
  "auth": {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "signature": "b1a927ec3..."
  }
}
# 【例8-6】
import time
import threading
from typing import Dict
from qwen_agent.agent import Agent
from qwen_agent.context import Message

# ========= 上下文容器类 =========
class ContextScope:
    def __init__(self):
        self.contexts: Dict[str, list] = {}

    def create(self, task_id: str):
        self.contexts[task_id] = []

    def append(self, task_id: str, message: Message):
        if task_id in self.contexts:
            self.contexts[task_id].append(message)

    def get(self, task_id: str) -> list:
        return self.contexts.get(task_id, [])

    def destroy(self, task_id: str):
        if task_id in self.contexts:
            del self.contexts[task_id]

# ========= 多任务Agent任务模拟 =========
class AgentTask(threading.Thread):
    def __init__(self, task_id: str, user_input: str, scope: ContextScope):
        super().__init__()
        self.task_id = task_id
        self.user_input = user_input
        self.scope = scope
        self.result = None

    def run(self):
        print(f"→ 启动任务[{self.task_id}]")
        self.scope.create(self.task_id)

        # 模拟输入并追加上下文
        self.scope.append(self.task_id, Message(role="user", content=self.user_input))

        # 构造代理并调用
        agent = Agent()
        history = self.scope.get(self.task_id)
        result = agent.chat(messages=history)
        self.scope.append(self.task_id, Message(role="assistant", content=result.content))

        self.result = result.content
        print(f"→ 任务[{self.task_id}]完成，结果：{self.result}")

# ========= 主流程 =========
if __name__ == "__main__":
    print("=== 启动多任务上下文隔离系统 ===")

    context_scope = ContextScope()

    tasks = [
        AgentTask("task_001", "请简要说明当前中国宏观经济趋势", context_scope),
        AgentTask("task_002", "总结2023年人工智能的发展方向", context_scope),
        AgentTask("task_003", "请写一段300字的自然语言处理技术简介", context_scope)
    ]

    for task in tasks:
        task.start()

    for task in tasks:
        task.join()

    # 汇总任务结果交由Qwen3.0整理
    combined_prompt = "以下是三个并行任务的输出，请统一生成一段内容总结：\n"
    for t in tasks:
        combined_prompt += f"\n【{t.task_id}】：{t.result}"

    final_agent = Agent()
    summary = final_agent.chat(messages=[Message(role="user", content=combined_prompt)])
    
    print("\n--- 汇总总结 ---")
    print(summary.content)




# 【例8-7】
import time
import threading
from typing import List
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

# ========= 共享状态类 =========
class SharedLog:
    def __init__(self):
        self.logs: List[str] = []
        self.lock = threading.Lock()

    def write_log(self, agent_name: str, message: str):
        with self.lock:
            entry = f"[{agent_name}]：{message}"
            print(f"→ 写入日志：{entry}")
            self.logs.append(entry)
            time.sleep(0.5)  # 模拟写入延迟

    def read_logs(self) -> str:
        return "\n".join(self.logs)

# ========= 子Agent定义 =========
class LoggingAgent(threading.Thread):
    def __init__(self, name: str, log: SharedLog, query: str):
        super().__init__()
        self.name = name
        self.shared_log = log
        self.query = query
        self.response = ""

    def run(self):
        print(f"→ {self.name}开始处理任务")
        agent = Agent()
        msg = Message(role="user", content=self.query)
        result = agent.chat(messages=[msg])
        self.response = result.content
        self.shared_log.write_log(self.name, self.response)
        print(f"→ {self.name}完成写入")

# ========= 主控系统 =========
if __name__ == "__main__":
    print("=== 启动状态同步与锁控制模拟系统 ===")

    shared_log = SharedLog()

    # 启动多个Agent模拟并发写入
    queries = [
        "请分析中国当前的宏观经济政策",
        "请概述当前全球通货膨胀形势",
        "请简要预测人工智能在2025年的发展趋势"
    ]
    agents = []
    for i, q in enumerate(queries):
        agent = LoggingAgent(f"Agent_{i+1}", shared_log, q)
        agents.append(agent)

    for agent in agents:
        agent.start()
    for agent in agents:
        agent.join()

    # 总结共享日志内容
    print("\n→ 所有Agent写入完成，准备总结：")
    print(shared_log.read_logs())

    summary_prompt = "以下是三个Agent的分析日志，请总结关键内容：\n" + shared_log.read_logs()
    summary_agent = Agent()
    final_result = summary_agent.chat(messages=[Message(role="user", content=summary_prompt)])

    print("\n--- Qwen3.0输出总结 ---")
    print(final_result.content)
