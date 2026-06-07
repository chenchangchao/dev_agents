# 【例11-4】多层级缓存与记忆 RAG Agent
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch11/src/11_4_memory_cache_rag_agent.py

import json
import uuid

from ch11_runtime import ask_llm, backend_name, data_path

KNOWLEDGE_FILE = data_path("sample_knowledge.txt")
SESSION_CACHE: dict[str, list[dict[str, str]]] = {}
MCP_CACHE: dict[str, list[dict[str, str]]] = {}

DEFAULT_KNOWLEDGE = [
    "Self-Attention机制通过Query、Key、Value计算序列内部不同位置之间的相关性。",
    "多头注意力机制通过多个注意力头并行捕捉不同子空间中的语义关系。",
    "Transformer使用注意力机制替代传统循环结构，提升并行计算效率。",
    "位置编码用于向模型注入序列中词元的位置信息。",
    "RoPE通过旋转位置编码将相对位置信息融入注意力计算。",
]


def ensure_knowledge_file() -> None:
    if not KNOWLEDGE_FILE.exists():
        KNOWLEDGE_FILE.write_text("\n\n".join(DEFAULT_KNOWLEDGE), encoding="utf-8")


def load_knowledge() -> list[str]:
    ensure_knowledge_file()
    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


def retrieve_knowledge(query: str, k: int = 2) -> list[str]:
    knowledge = load_knowledge()
    scored = []
    for chunk in knowledge:
        score = sum(1 for char in set(query) if char in chunk)
        scored.append((score, chunk))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [chunk for score, chunk in scored[:k] if score > 0] or knowledge[:k]


def cache_conversation(session_id: str, message: str, role: str = "user") -> None:
    SESSION_CACHE.setdefault(session_id, []).append({"role": role, "content": message})


def get_conversation_history(session_id: str) -> list[dict[str, str]]:
    return SESSION_CACHE.get(session_id, [])


class MemoryAgent:
    def __init__(self, session_id: str, agent_type: str = "default"):
        self.session_id = session_id
        self.agent_type = agent_type

    def chat(self, user_input: str) -> str:
        history = get_conversation_history(self.session_id)
        refs = retrieve_knowledge(user_input)
        prompt = (
            "请基于长期知识库和会话历史回答用户问题。\n\n"
            f"长期知识：\n{chr(10).join(refs)}\n\n"
            f"会话历史：\n{json.dumps(history, ensure_ascii=False)}\n\n"
            f"用户问题：{user_input}"
        )
        result = ask_llm(prompt, system="你是带长期知识和会话缓存的技术问答助手。", max_tokens=500, label=self.agent_type)
        cache_conversation(self.session_id, user_input, role="user")
        cache_conversation(self.session_id, result, role="assistant")
        return result


def cache_mcp_message(session_id: str, role: str, content: str) -> None:
    MCP_CACHE.setdefault(session_id, []).append({"role": role, "content": content})


def call_model(session_id: str, user_input: str, agent_type: str = "default") -> str:
    mcp_history = MCP_CACHE.get(session_id, [])
    refs = retrieve_knowledge(user_input)
    prompt = (
        "请根据MCP消息历史和检索知识回答。\n\n"
        f"MCP历史：{json.dumps(mcp_history, ensure_ascii=False)}\n"
        f"检索知识：{chr(10).join(refs)}\n"
        f"用户输入：{user_input}"
    )
    response = ask_llm(prompt, system="你是MCP协议层的知识问答助手。", max_tokens=500, label=agent_type)
    cache_mcp_message(session_id, "user", user_input)
    cache_mcp_message(session_id, "assistant", response)
    return response


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print(f"知识库路径：{KNOWLEDGE_FILE}")
    session_id = str(uuid.uuid4())
    agent = MemoryAgent(session_id=session_id, agent_type="memory_agent")

    print("Agent 回答:")
    print(agent.chat("请问Transformer中的Self-Attention机制是怎么工作的？"))
    print("\n继续对话回答:")
    print(agent.chat("它和多头注意力机制之间的关系是什么？"))

    mcp_session = str(uuid.uuid4())
    print("\n[Agent 回复]")
    print(call_model(mcp_session, "什么是位置编码？", agent_type="mcp_agent"))
    print("\n[继续回复]")
    print(call_model(mcp_session, "位置编码和RoPE有什么不同？", agent_type="mcp_agent"))


if __name__ == "__main__":
    main()
