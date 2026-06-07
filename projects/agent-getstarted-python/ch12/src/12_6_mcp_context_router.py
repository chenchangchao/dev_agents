# 【例12-6】MCP 上下文路由配置与调用
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch12/src/12_6_mcp_context_router.py

import uuid

from ch12_runtime import ask_with_context, backend_name


class MCPMessage:
    def __init__(self, role: str, content: str, call_type: str = "prompt"):
        self.role = role
        self.content = content
        self.call_type = call_type

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}


class MCPContext:
    def __init__(self, session_id: str | None = None):
        self.session_id = session_id or str(uuid.uuid4())
        self.history: list[MCPMessage] = []

    def add(self, role: str, content: str, call_type: str = "prompt") -> None:
        self.history.append(MCPMessage(role, content, call_type))

    def get_messages(self) -> list[dict[str, str]]:
        return [message.to_dict() for message in self.history]

    def get_last_intent(self) -> str:
        for message in reversed(self.history):
            if message.role == "user" and "查询" in message.content:
                return "qa"
            if message.role == "user" and "执行" in message.content:
                return "tool"
        return "dialog"


class MCPRouter:
    def route(self, context: MCPContext) -> str:
        intent = context.get_last_intent()
        print(f"[路由] 当前识别意图类型：{intent}")
        return ask_with_context(context.get_messages(), temperature=0.4, max_tokens=400, label=f"mcp_{intent}")


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    context = MCPContext()
    context.add("system", "你是一个知识丰富的智能助手")
    context.add("user", "请帮我查询一下2023年中国GDP是多少")
    context.add("assistant", "好的，我正在查询相关数据...")
    context.add("user", "此外，还请帮我执行一个统计脚本")
    print("[模型响应]", MCPRouter().route(context))


if __name__ == "__main__":
    main()
