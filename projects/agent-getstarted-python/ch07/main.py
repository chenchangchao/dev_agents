# 第7章 单智能体系统构建实战
# 【例7-1】
import uuid
import time
from enum import Enum
from typing import Dict, List
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

# ========= 状态定义 =========
class AgentState(Enum):
    INITIALIZED = "initialized"
    READY = "ready"
    RUNNING = "running"
    WAITING_TOOL = "waiting_tool"
    COMPLETED = "completed"
    FAILED = "failed"

# ========= 工具定义 =========
class ForecastTool(Tool):
    def __init__(self):
        super().__init__(name="deepseek_forecast", description="经济预测工具")

    def call(self, query: str) -> str:
        print("→ 调用 DeepSeek-V1 工具进行预测...")
        time.sleep(1)
        return f"[DeepSeek-V1] 预测结果：{query} 的增长预期为5.2%"

# ========= Agent任务结构 =========
class StatefulAgent:
    def __init__(self, tools: List[Tool]):
        self.task_id = str(uuid.uuid4())
        self.state = AgentState.INITIALIZED
        self.history: List[Dict] = []
        self.agent = Agent(tools=tools)
        self.tool = tools[0]

    def update_state(self, new_state: AgentState):
        print(f"→ 状态变更：{self.state.value} → {new_state.value}")
        self.state = new_state
        self.history.append({
            "timestamp": time.time(),
            "task_id": self.task_id,
            "state": self.state.value
        })

    def execute(self, system_prompt: str, user_question: str) -> str:
        try:
            self.update_state(AgentState.READY)

            full_prompt = f"""【系统提示】
{system_prompt}

【用户提问】
{user_question}

请结合相关经济数据工具完成结构化分析。"""

            self.update_state(AgentState.RUNNING)

            # 工具调用阶段
            self.update_state(AgentState.WAITING_TOOL)
            tool_result = self.tool.call(user_question)

            self.update_state(AgentState.RUNNING)

            # 构造模型最终输入
            merged_prompt = f"{full_prompt}\n【工具响应】\n{tool_result}\n请生成回答："
            result = self.agent.chat(messages=[Message(role="user", content=merged_prompt)])

            self.update_state(AgentState.COMPLETED)
            return result.content

        except Exception as e:
            self.update_state(AgentState.FAILED)
            return f"[ERROR] 任务执行失败：{str(e)}"

    def show_history(self):
        print("\n--- 状态变更日志 ---")
        for entry in self.history:
            t = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(entry["timestamp"]))
            print(f"{t} | 状态：{entry['state']}")

# ========= 主执行流程 =========
if __name__ == "__main__":
    print("=== 启动具备状态管理的智能体系统 ===")

    tool = ForecastTool()
    agent = StatefulAgent(tools=[tool])

    system_prompt = "你是一名宏观经济顾问，所有回答需结合当前CPI和GDP数据，表达严谨、简明。"
    user_question = "请预测2024年GDP增长趋势，并说明主要影响因素。"

    output = agent.execute(system_prompt, user_question)
    print("\n--- 模型响应输出 ---")
    print(output)

    agent.show_history()




# 【例7-2】
import json
from typing import Dict, List, Any
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

## 工具定义：经济预测工具
class EconomicForecastTool(Tool):
    def __init__(self):
        super().__init__(name="econ_forecast", description="基于区域和时间预测经济增长")

    def call(self, params: Dict[str, Any]) -> str:
        region = params.get("region", "中国")
        year = params.get("year", 2024)
        focus = params.get("focus", "GDP")

        return f"[DeepSeek-V1] 预测结果：{region}在{year}年的{focus}增长率预计为5.2%"

## 输入参数解析器
def parse_input(user_text: str) -> Dict[str, Any]:
    """
    简单的规则解析器：从文本中提取区域、年份和指标类型
    """
    region = "中国"
    year = 2024
    focus = "GDP"

    # 年份解析
    for token in user_text.split():
        if token.endswith("年") and token[:-1].isdigit():
            year = int(token[:-1])
    
    # 区域解析
    if "美国" in user_text:
        region = "美国"
    elif "欧盟" in user_text:
        region = "欧盟"

    # 指标关键词
    if "CPI" in user_text:
        focus = "CPI"
    elif "PMI" in user_text:
        focus = "PMI"

    return {
        "region": region,
        "year": year,
        "focus": focus
    }

## 日志打印函数
def print_parameters(params: Dict[str, Any]):
    print("\n--- 结构化参数 ---")
    for k, v in params.items():
        print(f"{k}: {v}")

## 主流程：Agent解析执行
if __name__ == "__main__":
    print("=== 启动输入参数解析与封装系统 ===")

    # 初始化工具与Agent
    forecast_tool = EconomicForecastTool()
    agent = Agent(tools=[forecast_tool])

    # 模拟用户自然语言输入
    user_input = "请预测2024年美国的GDP增长趋势"

    # 参数解析
    structured_params = parse_input(user_input)
    print_parameters(structured_params)

    # 传入工具执行
    tool_output = forecast_tool.call(structured_params)
    print("\n--- 工具输出结果 ---")
    print(tool_output)

    # 构造最终Prompt交由Qwen生成回答
    prompt = f"""请根据以下工具数据生成简洁经济解读：

工具输出：{tool_output}
问题原文：{user_input}
请用政策分析师的视角进行总结："""

    result = agent.chat(messages=[Message(role="user", content=prompt)])
    print("\n--- Qwen3.0 输出结果 ---")
    print(result.content)




# 【例7-3】
import time
import random
from typing import Dict, Any
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

## 工具定义（带异常处理）
class RiskyEconomicTool(Tool):
    def __init__(self):
        super().__init__(name="unstable_forecast", description="可能失败的经济预测工具")

    def call(self, inputs: Dict[str, Any]) -> str:
        try:
            region = inputs.get("region")
            if not region:
                raise ValueError("缺少region字段")
            
            # 模拟远程调用失败
            if random.random() < 0.5:
                raise ConnectionError("远程API调用失败：连接超时")

            # 正常预测结果
            return f"[DeepSeek-V1] 预测：{region} 2024年GDP增长预计为5.3%"
        
        except Exception as e:
            return f"[ERROR] 工具执行失败：{str(e)}"

## Agent包装器
class RobustAgent:
    def __init__(self, tool: Tool):
        self.agent = Agent(tools=[tool])
        self.tool = tool

    def process(self, user_input: str) -> str:
        region = "美国" if "美国" in user_input else "中国"

        # 工具调用
        tool_response = self.tool.call({"region": region})
        print(f"\n→ 工具返回结果：{tool_response}")

        if "[ERROR]" in tool_response:
            # 降级逻辑
            print("→ 进入降级流程，切换为模型直接生成")
            prompt = f"""注意：外部工具调用失败，请直接基于常识与历史数据，预测{region}在2024年的经济走势，并给出简明解释。"""
        else:
            # 正常逻辑
            prompt = f"""根据以下数据生成经济分析总结：
工具输出：{tool_response}
分析要求：语言简洁、观点明确、适合决策参考"""

        # 交由Qwen3.0生成
        result = self.agent.chat(messages=[Message(role="user", content=prompt)])
        return result.content

## 主流程运行
if __name__ == "__main__":
    print("=== 启动带异常处理的Agent系统 ===")

    tool = RiskyEconomicTool()
    agent = RobustAgent(tool=tool)

    # 模拟用户提问
    query = "请预测2024年美国的GDP增长趋势"
    response = agent.process(query)

    print("\n--- Qwen3.0 最终输出 ---")
    print(response)




# 【例7-4】
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

# 模拟工具模块
class FakeDataFetcher(Tool):
    def __init__(self):
        super().__init__(name="data_fetch", description="提供历史数据支持的模拟工具")

    def call(self, query: str) -> str:
        if "2023" in query:
            return "2023年GDP增长率为5.2%，主要受益于消费回暖和出口恢复"
        elif "2022" in query:
            return "2022年GDP增长为3.0%，疫情影响下内需低迷"
        else:
            return "暂无相关数据"

# 配置LangChain内存机制
memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# 构造Qwen Agent框架代理
class LangChainQwenAgent:
    def __init__(self, memory):
        self.memory = memory
        self.agent = Agent(tools=[FakeDataFetcher()])
    
    def ask(self, user_input: str) -> str:
        # 历史注入
        history_str = "\n".join([f"{m.content}" for m in self.memory.chat_memory.messages])
        full_prompt = f"""以下是最近对话历史：
{history_str}

用户现在的问题是：
{user_input}

请结合历史与知识进行回答："""

        # Qwen生成响应
        result = self.agent.chat(messages=[Message(role="user", content=full_prompt)])
        # 存储到LangChain记忆中
        self.memory.chat_memory.add_user_message(user_input)
        self.memory.chat_memory.add_ai_message(result.content)
        return result.content

# 初始化系统
if __name__ == "__main__":
    print("=== 启动结合LangChain记忆机制的Qwen3.0智能体 ===")
    bot = LangChainQwenAgent(memory=memory)

    # 多轮问答模拟
    print("\n[Round 1] 用户提问：2022年GDP是多少？")
    print(bot.ask("2022年GDP是多少？"))

    print("\n[Round 2] 用户提问：那2023年呢？")
    print(bot.ask("那2023年呢？"))

    print("\n[Round 3] 用户提问：你觉得2024年增长是否会更快？")
    print(bot.ask("你觉得2024年增长是否会更快？"))

    print("\n[Round 4] 用户追问：依据是什么？")
    print(bot.ask("依据是什么？"))




# 【例7-5】
import time
from typing import List, Dict
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

# 模拟工具模块
class EconomicSummaryTool(Tool):
    def __init__(self):
        super().__init__(name="econ_summary", description="生成结构化经济总结")

    def call(self, query: str) -> str:
        return f"[DeepSeek-V1] 分析完成：{query} 的GDP趋势良好。"

# 构建上下文段结构
def build_segment(role: str, content: str, segment_type: str, timestamp=None) -> Dict:
    return {
        "role": role,
        "content": content,
        "type": segment_type,
        "timestamp": timestamp or time.time(),
        "priority": 1 if segment_type == "system" else 0.5,
        "length": len(content) // 2  # 模拟token数
    }

# 动态剪辑策略
def dynamic_clip_context(segments: List[Dict], token_limit=150) -> str:
    # 按优先级与时间排序
    segments.sort(key=lambda x: (-x["priority"], -x["timestamp"]))
    
    total = 0
    clipped = []
    for seg in segments:
        if total + seg["length"] <= token_limit:
            clipped.append(seg)
            total += seg["length"]
    
    # 构造Prompt
    merged = ""
    for seg in clipped:
        merged += f"[{seg['role']}-{seg['type']}] {seg['content']}\n"
    return merged

# 主系统结构
if __name__ == "__main__":
    print("=== 启动具备上下文动态剪辑能力的Agent系统 ===")

    tool = EconomicSummaryTool()
    agent = Agent(tools=[tool])
    context_pool = []

    # 构建上下文段
    context_pool.append(build_segment("system", "你是资深经济顾问，语言需精准、简洁", "system"))
    context_pool.append(build_segment("user", "2022年GDP是多少？", "input"))
    context_pool.append(build_segment("assistant", "2022年增长为3.0%", "response"))
    context_pool.append(build_segment("user", "那2023年呢？", "input"))
    context_pool.append(build_segment("assistant", "2023年为5.2%", "response"))
    context_pool.append(build_segment("tool", tool.call("2024年预测"), "tool"))
    context_pool.append(build_segment("user", "2024年会不会更高？", "input"))
    context_pool.append(build_segment("assistant", "预计增长可达5.4%", "response"))
    context_pool.append(build_segment("user", "主要依据有哪些？", "input"))

    # 执行动态剪辑
    prompt = dynamic_clip_context(context_pool, token_limit=100)

    print("\n--- 剪辑后上下文结构 ---")
    print(prompt)

    # 模型调用
    result = agent.chat(messages=[Message(role="user", content=prompt)])
    print("\n--- Qwen3.0 响应输出 ---")
    print(result.content)
