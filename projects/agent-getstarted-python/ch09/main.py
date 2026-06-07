# 第9章 A2A协议：智能体之间的协作语言
# 【例9-1】
import time
import uuid
import json
import random
from datetime import datetime
from typing import Dict, Any

# === A2A消息格式定义 ===
def build_a2a_message(sender: str, receiver: str, msg_type: str, intent: str, context_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "id": f"msg-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}-{random.randint(100, 999)}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "from": sender,
        "to": receiver,
        "type": msg_type,
        "intent": intent,
        "context_id": context_id,
        "payload": payload,
        "auth": {
            "token": "demo-token-123456",
            "signature": "demo-signature-sha256"
        }
    }

# === Agent行为模拟 ===
class Agent:
    def __init__(self, name: str):
        self.name = name

    def handle_message(self, message: Dict[str, Any]) -> str:
        intent = message["intent"]
        task = message["payload"].get("task", "")
        params = message["payload"].get("params", {})
        print(f"\n[{self.name}] 接收到消息:")
        print(json.dumps(message, indent=2, ensure_ascii=False))

        if intent == "run_task" and task == "analyze_financial_trends":
            region = params.get("region", "未知")
            period = params.get("period", "未指定")
            result = f"分析完成：{region}地区{period}期间经济增长放缓，通胀风险可控。"
        elif intent == "query_weather":
            city = params.get("city", "未知")
            result = f"{city}天气晴朗，气温18-24℃。"
        else:
            result = "无法识别的任务指令"

        return result

# === 调度控制器 ===
class A2AController:
    def __init__(self):
        self.agents: Dict[str, Agent] = {}

    def register_agent(self, agent: Agent):
        self.agents[agent.name] = agent

    def dispatch(self, message: Dict[str, Any]) -> str:
        target = message["to"]
        if target in self.agents:
            return self.agents[target].handle_message(message)
        else:
            return f"目标Agent {target} 不存在"

# === 模拟执行 ===
if __name__ == "__main__":
    controller = A2AController()

    # 注册两个Agent
    planner = Agent("agent:planner")
    executor = Agent("agent:executor")
    weather = Agent("agent:weather")

    controller.register_agent(planner)
    controller.register_agent(executor)
    controller.register_agent(weather)

    # 生成一条结构化消息
    context_id = str(uuid.uuid4())

    msg1 = build_a2a_message(
        sender="agent:planner",
        receiver="agent:executor",
        msg_type="command",
        intent="run_task",
        context_id=context_id,
        payload={
            "task": "analyze_financial_trends",
            "params": {
                "region": "Asia",
                "period": "2024-Q4"
            }
        }
    )

    msg2 = build_a2a_message(
        sender="agent:planner",
        receiver="agent:weather",
        msg_type="request",
        intent="query_weather",
        context_id=context_id,
        payload={
            "task": "query_weather",
            "params": {
                "city": "上海"
            }
        }
    )

    print("\n--- 调度执行消息 1 ---")
    result1 = controller.dispatch(msg1)
    print(f"\n→ 执行结果：{result1}")

    print("\n--- 调度执行消息 2 ---")
    result2 = controller.dispatch(msg2)
    print(f"\n→ 执行结果：{result2}")




# 【例9-2】
import time
import uuid
import hashlib
import json
from typing import Dict, List, Any

# ========= 注册中心与认证系统 =========
class AgentRegistry:
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}

    def register_agent(self, name: str, token: str, signature: str, capabilities: List[str]):
        self.agents[name] = {
            "token": token,
            "signature": signature,
            "capabilities": capabilities
        }
        print(f"注册Agent：{name}，能力：{capabilities}")

    def is_authenticated(self, name: str, token: str, signature: str) -> bool:
        agent = self.agents.get(name)
        if not agent:
            return False
        return agent["token"] == token and agent["signature"] == signature

    def has_capability(self, name: str, task: str) -> bool:
        agent = self.agents.get(name)
        if not agent:
            return False
        return task in agent["capabilities"]

# ========= A2A消息构造函数 =========
def build_secure_message(sender: str, receiver: str, task: str, params: Dict[str, Any], token: str, signature: str) -> Dict[str, Any]:
    return {
        "id": str(uuid.uuid4()),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "from": sender,
        "to": receiver,
        "intent": "run_task",
        "payload": {
            "task": task,
            "params": params
        },
        "auth": {
            "token": token,
            "signature": signature
        }
    }

# ========= Agent处理类 =========
class Agent:
    def __init__(self, name: str):
        self.name = name

    def handle(self, message: Dict[str, Any]) -> str:
        task = message["payload"]["task"]
        if task == "text_summary":
            return "执行任务：文本摘要完成。"
        elif task == "translate_text":
            return "执行任务：翻译已完成。"
        return "未知任务。"

# ========= 模拟系统 =========
if __name__ == "__main__":
    # 初始化注册中心
    registry = AgentRegistry()

    # 模拟注册Agent
    def gen_token(agent_name): return hashlib.sha256(agent_name.encode()).hexdigest()

    registry.register_agent(
        name="agent:qwen3",
        token=gen_token("agent:qwen3"),
        signature="sig-qwen",
        capabilities=["text_summary", "translate_text"]
    )

    registry.register_agent(
        name="agent:deepseek",
        token=gen_token("agent:deepseek"),
        signature="sig-deep",
        capabilities=["translate_text"]
    )

    # 构造两个任务消息
    msg1 = build_secure_message(
        sender="agent:planner",
        receiver="agent:qwen3",
        task="text_summary",
        params={"text": "本项目聚焦于多智能体系统的通信机制"},
        token=gen_token("agent:qwen3"),
        signature="sig-qwen"
    )

    msg2 = build_secure_message(
        sender="agent:planner",
        receiver="agent:deepseek",
        task="text_summary",  # 注意：deepseek未声明该能力
        params={"text": "请将此句生成摘要"},
        token=gen_token("agent:deepseek"),
        signature="sig-deep"
    )

    # 任务调度逻辑
    def dispatch(agent_name: str, message: Dict[str, Any]):
        if not registry.is_authenticated(agent_name, message["auth"]["token"], message["auth"]["signature"]):
            return f"Agent {agent_name} 认证失败，拒绝执行。"
        if not registry.has_capability(agent_name, message["payload"]["task"]):
            return f"Agent {agent_name} 不具备执行 [{message['payload']['task']}] 的能力。"
        agent = Agent(agent_name)
        return agent.handle(message)

    print("\n--- 调度消息 1 ---")
    result1 = dispatch("agent:qwen3", msg1)
    print("→ 执行结果：", result1)

    print("\n--- 调度消息 2 ---")
    result2 = dispatch("agent:deepseek", msg2)
    print("→ 执行结果：", result2)




# 【例9-3】
from qwen_agent import Agent, Message, PromptEntry
from qwen_agent.tools import Tool
import uuid
import json
import time
from typing import Dict, Any, List

# ========= A2A通信结构 =========
class A2AMessage:
    def __init__(self, sender: str, receiver: str, intent: str, payload: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.sender = sender
        self.receiver = receiver
        self.intent = intent
        self.payload = payload

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "from": self.sender,
            "to": self.receiver,
            "intent": self.intent,
            "payload": self.payload
        }

# ========= MCP格式执行 =========
class ContextAwareAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.agent = Agent()

    def handle_message(self, message: A2AMessage) -> str:
        task = message.payload.get("task", "")
        content = message.payload.get("content", "")

        # 构建MCP提示结构
        prompt = [
            PromptEntry(role="system", content="你是一个具备高级理解能力的财经分析专家"),
            PromptEntry(role="user", content=f"请分析以下任务：{task}"),
            PromptEntry(role="memory", content=f"历史背景资料：{content}"),
        ]

        # 构建Message对象执行
        msg = Message(prompt=prompt)
        result = self.agent.chat(messages=[msg])
        return result.content

# ========= 主控调度系统 =========
class AgentOrchestrator:
    def __init__(self):
        self.agents: Dict[str, ContextAwareAgent] = {}

    def register_agent(self, agent_name: str, agent_obj: ContextAwareAgent):
        self.agents[agent_name] = agent_obj

    def dispatch_task(self, message: A2AMessage) -> str:
        target = message.receiver
        if target not in self.agents:
            return f"目标Agent [{target}] 不存在"
        result = self.agents[target].handle_message(message)
        return result

# ========= 执行逻辑入口 =========
if __name__ == "__main__":
    orchestrator = AgentOrchestrator()

    # 注册两个Agent
    agent1 = ContextAwareAgent(agent_name="agent:finance")
    agent2 = ContextAwareAgent(agent_name="agent:tech")

    orchestrator.register_agent("agent:finance", agent1)
    orchestrator.register_agent("agent:tech", agent2)

    # 构造两个任务
    task1 = A2AMessage(
        sender="agent:planner",
        receiver="agent:finance",
        intent="run_task",
        payload={
            "task": "分析2024年中国宏观经济走势",
            "content": "受益于内需恢复与出口回升，GDP预计实现5%以上增长。"
        }
    )

    task2 = A2AMessage(
        sender="agent:planner",
        receiver="agent:tech",
        intent="run_task",
        payload={
            "task": "预测2025年人工智能行业发展趋势",
            "content": "多模态模型与具身智能的结合将成为发展重点。"
        }
    )

    # 执行任务并输出结果
    print("\n--- 执行任务 1 ---")
    result1 = orchestrator.dispatch_task(task1)
    print("→ 结果：", result1)

    print("\n--- 执行任务 2 ---")
    result2 = orchestrator.dispatch_task(task2)
    print("→ 结果：", result2)




# 【例9-4】
from qwen_agent import Agent, Message, PromptEntry
import uuid
import time
from typing import Dict, Any

# ========= A2A消息结构 =========
class A2AMessage:
    def __init__(self, sender: str, receiver: str, msg_type: str, payload: Dict[str, Any]):
        self.id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.sender = sender
        self.receiver = receiver
        self.type = msg_type  # 'request' 或 'response'
        self.payload = payload

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "from": self.sender,
            "to": self.receiver,
            "type": self.type,
            "payload": self.payload
        }

# ========= Agent定义 =========
class RequestAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.agent = Agent()

    def build_request(self, task: str, question: str) -> A2AMessage:
        return A2AMessage(
            sender=self.agent_name,
            receiver="agent:responder",
            msg_type="request",
            payload={
                "task": task,
                "question": question
            }
        )

    def handle_response(self, response: A2AMessage):
        print(f"\n【{self.agent_name}】收到响应：")
        print(f"任务：{response.payload['task']}")
        print(f"回答：{response.payload['answer']}")

class ResponderAgent:
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.agent = Agent()

    def handle_request(self, message: A2AMessage) -> A2AMessage:
        task = message.payload["task"]
        question = message.payload["question"]

        prompt = [
            PromptEntry(role="system", content="你是一名专业问答机器人，回答简洁准确"),
            PromptEntry(role="user", content=question),
            PromptEntry(role="memory", content=f"任务类别：{task}")
        ]

        result = self.agent.chat(messages=[Message(prompt=prompt)])

        return A2AMessage(
            sender=self.agent_name,
            receiver=message.sender,
            msg_type="response",
            payload={
                "task": task,
                "answer": result.content
            }
        )

# ========= 调度控制器 =========
class Dispatcher:
    def __init__(self):
        self.request_agent = RequestAgent("agent:requester")
        self.responder_agent = ResponderAgent("agent:responder")

    def run(self):
        # 发出请求
        question = "当前欧洲的通货膨胀情况如何？"
        task = "财经问答"
        request_msg = self.request_agent.build_request(task, question)

        print(f"\n【{request_msg.sender}】发出请求：{question}")
        response_msg = self.responder_agent.handle_request(request_msg)

        # 处理响应
        self.request_agent.handle_response(response_msg)

if __name__ == "__main__":
    dispatcher = Dispatcher()
    dispatcher.run()




# 【例9-5】
from qwen_agent import Agent, Message, PromptEntry
import uuid
import time
from typing import Dict, List, Callable

# ========= 消息结构 =========
class PubSubMessage:
    def __init__(self, topic: str, content: str):
        self.id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.topic = topic
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "topic": self.topic,
            "content": self.content
        }

# ========= 订阅者定义 =========
class SubscriberAgent:
    def __init__(self, name: str, topic_filter: Callable[[str], bool]):
        self.name = name
        self.topic_filter = topic_filter
        self.agent = Agent()

    def on_message(self, message: PubSubMessage) -> str:
        if not self.topic_filter(message.topic):
            return f"{self.name}：忽略消息"

        prompt = [
            PromptEntry(role="system", content="你是一个专业财经分析Agent"),
            PromptEntry(role="user", content=f"请分析以下新闻：{message.content}"),
            PromptEntry(role="memory", content=f"分析主题：{message.topic}")
        ]
        response = self.agent.chat(messages=[Message(prompt=prompt)])
        return f"{self.name} 分析结果：{response.content}"

# ========= 发布者与调度器 =========
class PubSubSystem:
    def __init__(self):
        self.subscribers: List[SubscriberAgent] = []

    def register(self, agent: SubscriberAgent):
        self.subscribers.append(agent)

    def publish(self, message: PubSubMessage) -> List[str]:
        print(f"\n【广播】主题：{message.topic}")
        print(f"内容：{message.content}")
        results = []
        for agent in self.subscribers:
            result = agent.on_message(message)
            results.append(result)
        return results

# ========= 主程序 =========
if __name__ == "__main__":
    system = PubSubSystem()

    # 注册两个订阅者
    agent_qwen = SubscriberAgent("Qwen-Agent", lambda t: "财经" in t)
    agent_deepseek = SubscriberAgent("DeepSeek-Agent", lambda t: "AI" in t or "财经" in t)

    system.register(agent_qwen)
    system.register(agent_deepseek)

    # 发布广播消息
    message = PubSubMessage(
        topic="财经快讯",
        content="中国央行宣布降准50个基点以稳定宏观经济预期，股市应声上涨。"
    )

    results = system.publish(message)

    print("\n--- 响应结果 ---")
    for res in results:
        print(res)




# 【例9-6】
from qwen_agent import Agent, Message, PromptEntry
import uuid
import time
import random
from typing import Dict, List

# ========= 消息结构 =========
class BidMessage:
    def __init__(self, sender: str, task: str, proposal: str, score: float):
        self.id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.sender = sender
        self.task = task
        self.proposal = proposal
        self.score = score

    def to_dict(self) -> Dict[str, str]:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "task": self.task,
            "proposal": self.proposal,
            "score": self.score
        }

# ========= Agent定义 =========
class CompetitiveAgent:
    def __init__(self, name: str, expertise: str, capability_score: float):
        self.name = name
        self.expertise = expertise
        self.capability_score = capability_score
        self.agent = Agent()

    def bid_for_task(self, task_desc: str) -> BidMessage:
        prompt = [
            PromptEntry(role="system", content=f"你是一个在 {self.expertise} 领域具有丰富经验的Agent，请生成一份任务解决策略。"),
            PromptEntry(role="user", content=f"当前任务为：{task_desc}，请提供你的解决方案建议。"),
        ]
        result = self.agent.chat(messages=[Message(prompt=prompt)])
        return BidMessage(
            sender=self.name,
            task=task_desc,
            proposal=result.content,
            score=self.evaluate_score(result.content)
        )

    def evaluate_score(self, content: str) -> float:
        return round(self.capability_score + random.uniform(-0.3, 0.3), 2)

# ========= 调度器 =========
class TaskOrchestrator:
    def __init__(self):
        self.agents: List[CompetitiveAgent] = []

    def register(self, agent: CompetitiveAgent):
        self.agents.append(agent)

    def dispatch_task(self, task_desc: str) -> str:
        print(f"\n【主控Agent广播任务】：{task_desc}")
        bids = []
        for agent in self.agents:
            bid = agent.bid_for_task(task_desc)
            print(f"→ {bid.sender} 报价，能力评分：{bid.score}")
            bids.append(bid)

        # 选择评分最高者
        best_bid = max(bids, key=lambda x: x.score)
        print(f"\n【中标Agent】：{best_bid.sender}")
        print("中标方案：", best_bid.proposal)
        return best_bid.sender

# ========= 主程序 =========
if __name__ == "__main__":
    orchestrator = TaskOrchestrator()

    # 注册三个Agent
    orchestrator.register(CompetitiveAgent("Qwen-Agent", "宏观经济分析", 8.7))
    orchestrator.register(CompetitiveAgent("DeepSeek-Agent", "数据挖掘与预测", 8.4))
    orchestrator.register(CompetitiveAgent("Coze-Agent", "财经舆情解读", 8.6))

    # 广播任务并评标
    selected = orchestrator.dispatch_task("请分析人民币汇率波动对外贸企业的影响，并提出缓解建议。")
