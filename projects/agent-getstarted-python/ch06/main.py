# 第6章 MCP协议：模型上下文通信标准
# 【例6-1】
import json
import requests
from typing import List, Dict
from qwen_agent.agent import Agent
from qwen_agent.tools import Tool
from qwen_agent.context import Message

## 上下文段结构体构建函数
def build_context_segment(role: str, content: str, segment_type: str, task_id: str, meta: Dict = None) -> Dict:
    """构建标准上下文段结构"""
    return {
        "role": role,  # 角色，如 user / system / tool / model
        "content": content,  # 上下文内容
        "type": segment_type,  # 段落类型，如 input / response / observation
        "task_id": task_id,  # 所属任务标识
        "meta": meta or {}  # 附加元信息，如时间戳、状态码等
    }

## DeepSeek-V1模拟调用接口
def call_deepseek_v1(prompt: str) -> str:
    # 示例：调用DeepSeek-V1模型 API（此处用虚拟请求替代）
    print("调用DeepSeek-V1生成中……")
    return f"[DeepSeek] 回答：根据提示'{prompt}'生成的答案。"

## 结构化上下文管理工具
class StructuredContextQueryTool(Tool):
    def __init__(self, segments: List[Dict]):
        super().__init__(name="structured_context_query", description="结构化上下文段解读工具")
        self.segments = segments

    def call(self, query: str) -> str:
        # 仅抽取用户历史输入与模型回答组成上下文摘要
        relevant = [seg for seg in self.segments if seg["type"] in ["input", "response"]]
        context_summary = "\n".join([f"{seg['role']}: {seg['content']}" for seg in relevant[-4:]])
        return f"【对话上下文】\n{context_summary}\n【用户问题】\n{query}"

## 主流程
if __name__ == "__main__":
    # 构建上下文段
    context_segments = [
        build_context_segment("user", "我想了解2023年中国的财政支出情况。", "input", task_id="t001"),
        build_context_segment("model", "2023年中国财政支出约为26万亿元，主要用于民生保障、教育和基础设施建设。", "response", task_id="t001"),
        build_context_segment("user", "那2022年与2023年相比增长了多少？", "input", task_id="t001")
    ]

    # 构建结构化上下文摘要工具
    context_tool = StructuredContextQueryTool(segments=context_segments)

    # Qwen3.0 Agent调用
    agent = Agent(tools=[context_tool])

    # 用户最终提问
    query = "是否存在财政支出快速增长的趋势？"
    structured_prompt = context_tool.call(query)

    # 使用Qwen3.0生成
    qwen_response = agent.chat(messages=[Message(role="user", content=structured_prompt)])

    # 调用DeepSeek-V1补充另一种视角（假设处理结构化更强）
    deepseek_response = call_deepseek_v1(prompt=structured_prompt)

    # 合并响应
    print("\n--- Qwen3.0 Agent 回答 ---")
    print(qwen_response.content)

    print("\n--- DeepSeek-V1 辅助回答 ---")
    print(deepseek_response)

    # 增加新段落
    context_segments.append(build_context_segment("model", qwen_response.content, "response", "t001"))
    context_segments.append(build_context_segment("model", deepseek_response, "observation", "t001", meta={"source": "deepseek"}))

    # 打印结构化上下文记录
    print("\n--- 当前上下文段结构 ---")
    print(json.dumps(context_segments, ensure_ascii=False, indent=2))




# 【例6-2】
import json
from typing import List, Dict
from qwen_agent.agent import Agent
from qwen_agent.tools import Tool
from qwen_agent.context import Message

# ========= 上下文段构造函数 =========
def build_segment(role: str, content: str, seg_type: str, task_id: str, meta: Dict = None) -> Dict:
    """构建结构化上下文段"""
    return {
        "role": role,
        "content": content,
        "type": seg_type,
        "task_id": task_id,
        "meta": meta or {}
    }

# ========= 模拟DeepSeek工具调用 =========
def call_deepseek_tool(prompt: str) -> str:
    print("调用DeepSeek-V1模型工具中……")
    return f"[DeepSeek Tool] 对于'{prompt}'的结构化回答：预计2024年财政收入将增长5%左右，主要来源为消费税和企业所得税。"

# ========= 工具响应段模拟 =========
class FiscalForecastTool(Tool):
    def __init__(self):
        super().__init__(name="fiscal_forecast", description="模拟财政预测工具")

    def call(self, query: str) -> str:
        return call_deepseek_tool(query)

# ========= 主流程 =========
if __name__ == "__main__":
    segments: List[Dict] = []

    # 系统提示段：指定风格与角色设定
    system_prompt = "你是一位熟悉中国经济政策的政府咨询顾问，回答需专业、简明、有依据。"
    segments.append(build_segment("system", system_prompt, "system", task_id="t001"))

    # 记忆段：注入用户长期偏好
    memory_info = "用户偏好获取简明扼要的政策摘要，避免冗长表述。"
    segments.append(build_segment("memory", memory_info, "memory", task_id="t001"))

    # 用户输入段
    user_question = "请预测2024年中国的财政收入趋势，并说明主要增长来源。"
    segments.append(build_segment("user", user_question, "input", task_id="t001"))

    # 工具段：模拟调用结构化工具返回结果
    tool = FiscalForecastTool()
    tool_response = tool.call(user_question)
    segments.append(build_segment("tool", tool_response, "tool", task_id="t001", meta={"tool": "deepseek-v1"}))

    # 构造Agent上下文
    context_summary = "\n".join(
        [f"[{s['type'].upper()}] {s['role']}: {s['content']}" for s in segments]
    )

    # 创建Qwen Agent并发送消息
    agent = Agent(tools=[tool])
    final_prompt = f"以下是结构化上下文段，请基于这些内容给出准确回答：\n{context_summary}"
    response = agent.chat(messages=[Message(role="user", content=final_prompt)])

    # 输出内容
    print("\n--- Qwen3.0 Agent 回答 ---")
    print(response.content)

    print("\n--- 当前上下文段结构 ---")
    print(json.dumps(segments, ensure_ascii=False, indent=2))




# 【例6-3】
import json
from typing import List, Dict
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

# ========= 路由规则定义 =========
def determine_route(user_input: str) -> str:
    """根据关键词判断信息应路由到哪个模块"""
    if "政策" in user_input or "法律" in user_input:
        return "qwen"
    elif "数据" in user_input or "预测" in user_input:
        return "deepseek"
    elif "调用" in user_input and "工具" in user_input:
        return "tool"
    else:
        return "default"

# ========= 工具模块定义 =========
class DataSummaryTool(Tool):
    def __init__(self):
        super().__init__(name="data_summary", description="结构化数据总结工具")

    def call(self, query: str) -> str:
        return "[Tool] 已根据结构化数据对2023年财政执行情况完成分析，总支出同比增长6.8%。"

# ========= 模拟DeepSeek-V1调用 =========
def call_deepseek_module(prompt: str) -> str:
    print("→ 已路由至 DeepSeek-V1 处理数据类任务。")
    return f"[DeepSeek] 回答：针对'{prompt}'，预测2024年经济增长为5%-5.5%。"

# ========= 构造Agent上下文消息 =========
def build_context_message(role: str, content: str) -> Message:
    return Message(role=role, content=content)

# ========= 主流程入口 =========
if __name__ == "__main__":
    # 初始化模型与工具
    tool = DataSummaryTool()
    agent = Agent(tools=[tool])

    # 模拟用户请求
    user_input = "请调用工具分析2023年的财政数据，并预测2024年中国GDP的增长情况，还要说明近期的主要财政政策。"

    # 拆分用户指令
    sub_tasks = [
        "分析2023年的财政数据",  # route → tool
        "预测2024年中国GDP的增长情况",  # route → deepseek
        "说明近期的主要财政政策"  # route → qwen
    ]

    # 构建任务调度与执行路由
    responses: List[str] = []
    for task in sub_tasks:
        route = determine_route(task)

        if route == "qwen":
            print(f"→ 已路由至 Qwen3.0 处理政策类任务。任务内容：{task}")
            prompt = f"请作为政策专家，简要说明：{task}"
            result = agent.chat(messages=[build_context_message("user", prompt)])
            responses.append(result.content)

        elif route == "deepseek":
            result = call_deepseek_module(task)
            responses.append(result)

        elif route == "tool":
            result = tool.call(task)
            responses.append(result)

        else:
            print("→ 无匹配路由，使用默认模型。")
            prompt = f"默认处理：{task}"
            result = agent.chat(messages=[build_context_message("user", prompt)])
            responses.append(result.content)

    # 输出合并结果
    print("\n--- 多模块合并响应 ---")
    for idx, res in enumerate(responses, 1):
        print(f"[子任务{idx}] {res}")




# 【例6-4】
import json
from typing import List, Dict
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

## 模拟模型调用函数定义
def call_qwen3(prompt: str) -> str:
    print("→ Qwen3.0入口已触发。")
    return f"[Qwen3.0] 回答：针对'{prompt}'，已完成语言生成与政策解读。"

def call_deepseek(prompt: str) -> str:
    print("→ DeepSeek-V1入口已触发。")
    return f"[DeepSeek-V1] 分析：关于'{prompt}'的预测已完成，增长预期为5.3%。"

## 分支决策逻辑器
def route_decision(query: str) -> str:
    """根据语义关键词决策模型入口"""
    if any(k in query for k in ["政策", "法规", "概述", "总结"]):
        return "qwen"
    elif any(k in query for k in ["预测", "趋势", "估计", "数据"]):
        return "deepseek"
    elif any(k in query for k in ["综合分析", "多维度", "对比"]):
        return "hybrid"
    else:
        return "default"

## 工具模块示例（用于辅助对比任务）
class HybridAnalysisTool(Tool):
    def __init__(self):
        super().__init__(name="hybrid_analysis", description="多模型结果融合工具")

    def call(self, inputs: Dict[str, str]) -> str:
        qwen_result = call_qwen3(inputs["policy"])
        deepseek_result = call_deepseek(inputs["forecast"])
        return f"[组合分析] 综合政策视角：{qwen_result}\n数据预测视角：{deepseek_result}"

## 主流程入口
if __name__ == "__main__":
    print("=== 启动模型入口决策示例系统 ===")

    # 模拟用户提问集
    user_queries = [
        "请概述近期财政补贴政策的主要方向",
        "请预测2024年GDP增长趋势",
        "请对近期财政补贴政策与经济增长趋势进行多维度综合分析"
    ]

    # 初始化Qwen Agent
    tool = HybridAnalysisTool()
    agent = Agent(tools=[tool])

    # 按任务决策执行分支
    for idx, query in enumerate(user_queries, 1):
        print(f"\n--- 处理第{idx}条请求 ---")
        decision = route_decision(query)

        if decision == "qwen":
            result = call_qwen3(query)

        elif decision == "deepseek":
            result = call_deepseek(query)

        elif decision == "hybrid":
            result = tool.call(inputs={
                "policy": "财政补贴政策分析",
                "forecast": "2024年经济趋势"
            })

        else:
            print("→ 默认模型入口激活")
            result = agent.chat(messages=[Message(role="user", content=query)]).content

        print(f"\n结果输出：\n{result}")




# 【例6-5】
import os
import json
import uuid
from datetime import datetime
from typing import Dict, List
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

## 日志持久化路径设置
LOG_FILE = "agent_context_log.jsonl"

## 工具定义：模拟DeepSeek工具
class DeepSeekPredictor(Tool):
    def __init__(self):
        super().__init__(name="deepseek_predict", description="用于经济数据预测的模拟工具")

    def call(self, query: str) -> str:
        print("→ 已调用 DeepSeek-V1 预测模块")
        return f"[DeepSeek-V1] 针对'{query}'预测2024年GDP增长率为5.2%。"

## 上下文日志段生成函数
def create_log_segment(role: str, content: str, segment_type: str, model: str, task_id: str) -> Dict:
    return {
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now().isoformat(),
        "task_id": task_id,
        "role": role,
        "type": segment_type,
        "model": model,
        "content": content
    }

## 持久化日志写入函数
def persist_segment(segment: Dict):
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(segment, ensure_ascii=False) + "\n")

## 主流程执行
if __name__ == "__main__":
    print("=== 持久化上下文日志系统启动 ===")

    # 初始化Agent和工具
    tool = DeepSeekPredictor()
    agent = Agent(tools=[tool])
    task_id = "t006"

    # 1. 系统提示段
    system_prompt = "你是中国国家财政专家助手，所有回答需简洁、严谨、有数据支撑。"
    seg = create_log_segment("system", system_prompt, "system_prompt", "qwen-3.0", task_id)
    persist_segment(seg)

    # 2. 用户提问段
    user_input = "请预测2024年中国GDP增长，并说明依据。"
    seg = create_log_segment("user", user_input, "input", "user", task_id)
    persist_segment(seg)

    # 3. 工具调用段（模拟DeepSeek预测）
    tool_output = tool.call(user_input)
    seg = create_log_segment("tool", tool_output, "tool_output", "deepseek-v1", task_id)
    persist_segment(seg)

    # 4. 最终生成回答段（由Qwen3.0生成）
    final_prompt = f"""系统提示：{system_prompt}
工具信息：{tool_output}
用户提问：{user_input}
请基于以上信息进行严谨回答："""
    result = agent.chat(messages=[Message(role="user", content=final_prompt)])
    seg = create_log_segment("assistant", result.content, "response", "qwen-3.0", task_id)
    persist_segment(seg)

    # 输出结果
    print("\n--- 最终回答输出 ---")
    print(result.content)

    # 展示持久化日志（最近5条）
    print("\n--- 最新上下文日志（展示最后5条） ---")
    with open(LOG_FILE, "r", encoding="utf-8") as f:
        lines = f.readlines()[-5:]
        for line in lines:
            print(json.loads(line))




# 【例6-6】
import json
import hashlib
import os
from typing import Dict, List
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

# 缓存路径配置
CACHE_FILE = "prompt_cache.json"

# 工具定义：DeepSeek模拟分析工具
class EconomicPredictor(Tool):
    def __init__(self):
        super().__init__(name="econ_predictor", description="经济预测模拟工具")

    def call(self, query: str) -> str:
        print("→ DeepSeek-V1调用：分析完成")
        return f"[DeepSeek-V1] 预测结果：{query} 预计增长率为5.3%。"

# 生成任务唯一hash值
def hash_prompt(prompt: str) -> str:
    return hashlib.md5(prompt.encode("utf-8")).hexdigest()

# 缓存读取函数
def load_prompt_cache() -> Dict[str, str]:
    if not os.path.exists(CACHE_FILE):
        return {}
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

# 缓存写入函数
def save_prompt_cache(cache: Dict[str, str]):
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

# 缓存匹配与命中逻辑
def check_prompt_cache(prompt: str, cache: Dict[str, str]) -> str:
    h = hash_prompt(prompt)
    return cache.get(h, "")

# 主流程入口
if __name__ == "__main__":
    print("=== Prompt缓存与快速回放机制测试系统 ===")

    # 初始化模型与工具
    agent = Agent(tools=[EconomicPredictor()])
    prompt_cache = load_prompt_cache()

    # 构造复杂任务prompt
    user_query = "请结合当前宏观经济环境预测2024年GDP走势，并指出主要影响因素"
    system_prompt = "你是一位宏观经济分析师，回答应包含数据推理、趋势预测与政策建议"
    tool_output = "[Tool] 当前CPI指数为2.1%，PMI为51.4，消费同比上升6.2%"

    full_prompt = f"""【系统提示】
{system_prompt}
【外部工具输出】
{tool_output}
【用户提问】
{user_query}
请基于上述内容生成结构化分析结果"""

    prompt_hash = hash_prompt(full_prompt)
    print(f"→ 任务哈希ID：{prompt_hash}")

    # 检查缓存是否命中
    cached_output = check_prompt_cache(full_prompt, prompt_cache)
    if cached_output:
        print("\n--- 缓存命中，直接回放 ---")
        print(cached_output)
    else:
        print("\n--- 未命中缓存，调用模型生成 ---")
        # 模拟生成
        result = agent.chat(messages=[Message(role="user", content=full_prompt)])
        generated_answer = result.content
        print(generated_answer)

        # 写入缓存
        prompt_cache[prompt_hash] = generated_answer
        save_prompt_cache(prompt_cache)

    # 查看当前缓存总数
    print(f"\n当前缓存条目数量：{len(prompt_cache)}")




# 【例6-7】
import json
import time
from typing import List, Dict
from qwen_agent.agent import Agent
from qwen_agent.context import Message
from qwen_agent.tools import Tool

## 定义结构化上下文段结构
def build_segment(role: str, content: str, segment_type: str, timestamp: float = None) -> Dict:
    return {
        "role": role,
        "content": content,
        "type": segment_type,
        "timestamp": timestamp or time.time()
    }

## 模拟DeepSeek工具模块
class MacroForecastTool(Tool):
    def __init__(self):
        super().__init__(name="macro_forecast", description="经济趋势分析工具")

    def call(self, query: str) -> str:
        print("→ DeepSeek-V1 调用中...")
        return f"[DeepSeek-V1] 分析：{query} 增长趋势预期良好，2024年预计5.3%。"

## 上下文合并策略
def dynamic_merge_context(segments: List[Dict], task_type: str) -> str:
    # 策略：优先保留系统提示、工具输出，其次为最近2条用户+模型交互
    selected = []

    # 系统提示保留
    system_segments = [s for s in segments if s["type"] == "system"]
    selected.extend(system_segments)

    # 工具段保留最新1条
    tool_segments = sorted([s for s in segments if s["type"] == "tool"], key=lambda x: -x["timestamp"])
    if tool_segments:
        selected.append(tool_segments[0])

    # 最近交互保留2轮
    history = sorted([s for s in segments if s["type"] in ["input", "response"]], key=lambda x: -x["timestamp"])
    selected.extend(history[:4])

    # 构造合并Prompt
    merged = ""
    for seg in selected:
        prefix = f"[{seg['role'].upper()}-{seg['type']}]"
        merged += f"{prefix} {seg['content']}\n"

    return merged

## 主流程执行
if __name__ == "__main__":
    print("=== 启动动态上下文合并策略系统 ===")

    # 初始化上下文池与工具
    context_pool: List[Dict] = []
    tool = MacroForecastTool()
    agent = Agent(tools=[tool])

    # 生成系统提示段
    context_pool.append(build_segment("system", "你是一位国家政策与宏观经济专家，语言要严谨、简洁、具数据支撑", "system"))

    # 模拟历史对话与工具输出
    context_pool.append(build_segment("user", "2022年GDP增长率是多少？", "input"))
    context_pool.append(build_segment("assistant", "2022年中国GDP增长为3.0%。", "response"))

    context_pool.append(build_segment("user", "2023年预计呢？", "input"))
    context_pool.append(build_segment("assistant", "2023年GDP增长约为5.2%。", "response"))

    tool_output = tool.call("2024年GDP预测")
    context_pool.append(build_segment("tool", tool_output, "tool"))

    # 用户当前提问
    new_question = "请基于历史数据与趋势，判断2024年GDP是否可能超过5.5%？"
    context_pool.append(build_segment("user", new_question, "input"))

    # 合并上下文并调用模型
    final_prompt = dynamic_merge_context(context_pool, task_type="forecast")
    print("\n--- 合并后的Prompt结构 ---")
    print(final_prompt)

    # 交给Qwen3.0处理
    result = agent.chat(messages=[Message(role="user", content=final_prompt)])
    print("\n--- Qwen3.0 输出结果 ---")
    print(result.content)
