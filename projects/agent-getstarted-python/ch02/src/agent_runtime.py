import json
import re
from pathlib import Path
from typing import Any

CH02_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = CH02_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def data_path(*parts: str) -> Path:
    path = DATA_DIR.joinpath(*parts)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def memory_path(*parts: str) -> Path:
    return data_path("memory", *parts)


def cleanup_path(*parts: str) -> Path:
    return data_path("memory_cleanup", *parts)


def safe_workspace_path(filepath: str) -> Path:
    raw = Path(filepath)
    if raw.is_absolute():
        path = raw
    else:
        path = data_path("files", str(raw))
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class BaseTool:
    name: str = ""

    def __init__(self, *args, **kwargs):
        if not self.name:
            self.name = _tool_name(self.__class__.__name__)

    def call(self, params: str | dict | None = None, **kwargs):
        return self.run(parse_params(params))

    def run(self, params: dict) -> str:
        raise NotImplementedError


Tool = BaseTool


class BaseMemory:
    pass


class SimpleMemory(BaseMemory):
    def __init__(self):
        self.messages = []

    def add_message(self, message: dict):
        self.messages.append(message)

    def get(self):
        return self.messages


def llm_config(model: str = "local-rule-router", **generate_cfg):
    return {
        "model": model,
        "generate_cfg": generate_cfg,
    }


def parse_params(params: str | dict | None) -> dict:
    if params is None:
        return {}
    if isinstance(params, dict):
        return params
    try:
        return json.loads(params)
    except json.JSONDecodeError:
        return {}


class LocalAgent:
    """A tiny local rule-based Agent used to replace qwen_agent for chapter 2 demos."""

    def __init__(self, name: str, tools=None, memory=None, system_message: str = "", **kwargs):
        self.name = name
        self.tools = tools or []
        self.memory = memory
        self.system_message = system_message
        self.tool_map = {tool.name: tool for tool in self.tools}

    def chat(self, user_input: str) -> str:
        if self.memory and hasattr(self.memory, "add_message"):
            self.memory.add_message({"role": "user", "content": user_input})

        answer = self._route(user_input)

        if self.memory and hasattr(self.memory, "add_message"):
            self.memory.add_message({"role": "assistant", "content": answer})
        return answer

    def _route(self, user_input: str) -> str:
        if "系统检查" in user_input and "startup_check" in self.tool_map:
            return self.tool_map["startup_check"].call({})
        if ("几点" in user_input or "时间" in user_input) and "get_time" in self.tool_map:
            return self.tool_map["get_time"].call({})
        if "会议" in user_input:
            if "主题" in user_input and "set_topic" in self.tool_map:
                return self.tool_map["set_topic"].call({"topic": user_input})
            if "时间" in user_input and "set_time" in self.tool_map:
                return self.tool_map["set_time"].call({"time": user_input})
            if ("参加" in user_input or "与会" in user_input) and "set_attendees" in self.tool_map:
                return self.tool_map["set_attendees"].call({"attendees": user_input})
        if "记录" in user_input and "log_user_input" in self.tool_map:
            return self.tool_map["log_user_input"].call({"content": user_input})
        if "天气" in user_input and "weather_api" in self.tool_map:
            return self.tool_map["weather_api"].call({"city": extract_city(user_input)})
        if ("查询" in user_input or "列表" in user_input) and "query_employee" in self.tool_map:
            return self.tool_map["query_employee"].call({"name": extract_name(user_input)})
        if ("添加" in user_input or "员工" in user_input) and "add_employee" in self.tool_map:
            return self.tool_map["add_employee"].call(extract_employee(user_input))
        if "读取" in user_input and "read_file" in self.tool_map:
            return self.tool_map["read_file"].call({"filepath": extract_filepath(user_input)})
        if "写入" in user_input and "write_file" in self.tool_map:
            return self.tool_map["write_file"].call(extract_write_file(user_input))
        if ("运行" in user_input or "执行" in user_input) and "exec_code" in self.tool_map:
            return self.tool_map["exec_code"].call({"code": user_input.split("：", 1)[-1]})
        if ("消费" in user_input or "花了" in user_input or "买" in user_input) and "add_record" in self.tool_map:
            return self.tool_map["add_record"].call(extract_record(user_input))
        if "task" in self.tool_map or "任务" in user_input:
            tool = self.tool_map.get("task")
            if tool:
                return tool.call({"task": user_input})
        return f"{self.name}收到：{user_input}"


def _tool_name(class_name: str) -> str:
    name = re.sub(r"Tool$", "", class_name)
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


def extract_city(text: str) -> str:
    for city in ["北京", "上海", "广州", "东京"]:
        if city in text:
            return city
    return text.strip()


def extract_name(text: str) -> str:
    match = re.search(r"姓名是?([\u4e00-\u9fa5]{2,4})", text)
    return match.group(1) if match else ""


def extract_employee(text: str) -> dict[str, Any]:
    return {
        "name": extract_name(text) or "王小明",
        "department": "技术部" if "技术" in text else "",
        "title": "高级工程师" if "高级工程师" in text else "",
    }


def extract_filepath(text: str) -> str:
    match = re.search(r"([A-Za-z0-9_./-]+\.txt)", text)
    return match.group(1) if match else "test.txt"


def extract_write_file(text: str) -> dict[str, str]:
    filepath = extract_filepath(text)
    content = text.split("：", 1)[-1] if "：" in text else text
    return {"filepath": filepath, "content": content}


def extract_record(text: str) -> dict[str, str]:
    amount_match = re.search(r"(\d+(?:\.\d+)?)", text)
    category = "消费"
    if "买菜" in text:
        category = "买菜"
    return {
        "category": category,
        "amount": amount_match.group(1) if amount_match else "0",
    }
