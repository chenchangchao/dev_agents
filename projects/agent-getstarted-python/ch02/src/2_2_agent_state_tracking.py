# 【例2-2】
# agent_state_tracking.py

from agent_runtime import BaseTool, LocalAgent as Agent, SimpleMemory, Tool, llm_config, parse_params

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
