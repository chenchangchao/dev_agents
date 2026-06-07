# 【例2-5】
# agent_web_api_weather.py

import requests
import json
from agent_runtime import BaseTool, LocalAgent as Agent,  llm_config

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
