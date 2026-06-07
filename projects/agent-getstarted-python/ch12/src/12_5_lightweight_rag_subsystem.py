# 【例12-5】轻量 RAG 检索子系统
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch12/src/12_5_lightweight_rag_subsystem.py

from ch12_runtime import ask_llm, backend_name, data_path, ensure_text_file

KNOWLEDGE_PATH = data_path("ai_knowledge.txt")
DEFAULT_KNOWLEDGE = [
    "人工智能是研究如何让机器模拟、延伸和扩展人类智能的技术领域。",
    "机器学习是人工智能的重要分支，强调从数据中学习规律。",
    "深度学习通常使用多层神经网络处理复杂模式识别任务。",
]


def build_knowledge_from_file() -> list[str]:
    ensure_text_file(KNOWLEDGE_PATH, DEFAULT_KNOWLEDGE)
    text = KNOWLEDGE_PATH.read_text(encoding="utf-8")
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


def retrieve(query: str, knowledge: list[str], k: int = 2) -> list[str]:
    scored = []
    for chunk in knowledge:
        score = sum(1 for char in set(query) if char in chunk)
        scored.append((score, chunk))
    scored.sort(reverse=True, key=lambda item: item[0])
    return [chunk for score, chunk in scored[:k] if score > 0] or knowledge[:k]


class RAGAgent:
    def __init__(self, knowledge: list[str]):
        self.knowledge = knowledge
        self.history: list[tuple[str, str]] = []

    def ask(self, query: str) -> str:
        refs = retrieve(query, self.knowledge)
        prompt = (
            "请基于参考资料回答用户问题。如果资料不足，请说明可能需要补充信息。\n\n"
            f"参考资料：\n{chr(10).join(refs)}\n\n"
            f"历史对话：{self.history}\n"
            f"用户问题：{query}"
        )
        result = ask_llm(prompt, system="你是轻量RAG问答助手。", temperature=0.3, max_tokens=500, label="rag_agent")
        self.history.append((query, result))
        return result


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print(f"知识库路径：{KNOWLEDGE_PATH}")
    agent = RAGAgent(build_knowledge_from_file())
    for index, query in enumerate(["请简要介绍什么是人工智能？", "人工智能和机器学习有什么区别？", "深度学习属于人工智能吗？"], start=1):
        print(f"\n问题{index}：{query}")
        print("回答：", agent.ask(query))


if __name__ == "__main__":
    main()
