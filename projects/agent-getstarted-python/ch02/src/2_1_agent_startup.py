# 【例2-1】
# agent_startup.py

import logging
import datetime
from typing import List
from agent_runtime import BaseTool, LocalAgent as Agent, Tool, data_path, llm_config

Tool = BaseTool
ChatMessage = dict

# 配置日志
logging.basicConfig(
    filename=data_path("logs", "agent_startup.log"),
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
