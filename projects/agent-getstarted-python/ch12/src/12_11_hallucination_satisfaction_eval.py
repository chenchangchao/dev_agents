"""例12-11：幻觉率与满意度评估。

运行方式：
cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
python3 ch12/src/12_11_hallucination_satisfaction_eval.py
LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=qwen3:1.7b python3 ch12/src/12_11_hallucination_satisfaction_eval.py
"""

from __future__ import annotations

import difflib
import random
from dataclasses import dataclass

from ch12_runtime import ask_llm, backend_name


@dataclass
class EvalCase:
    question: str
    reference: str


EVAL_SET = [
    EvalCase("请问2022年中国的GDP是多少？", "2022年中国GDP约为121万亿元人民币"),
    EvalCase("地球上最大的哺乳动物是什么？", "地球上最大的哺乳动物是蓝鲸"),
    EvalCase("爱因斯坦是哪一年获得诺贝尔奖的？", "爱因斯坦于1921年获得诺贝尔物理学奖"),
    EvalCase("Transformer模型由谁在什么论文中提出？", "Transformer由Vaswani等人在2017年论文Attention Is All You Need中提出"),
    EvalCase("请列出中国的四大发明", "中国四大发明包括造纸术、指南针、火药和印刷术"),
]


class EvalAgent:
    def ask(self, text: str) -> str:
        return ask_llm(
            text,
            system="你是严谨的事实问答助手。不了解时应明确说明不确定，不要编造。",
            temperature=0,
            max_tokens=220,
            label="hallucination-eval",
        )


def hallucination_detect(generated: str, reference: str) -> tuple[bool, float]:
    similarity = difflib.SequenceMatcher(None, generated, reference).ratio()
    return similarity < 0.35, similarity


def user_score(generated: str, similarity: float) -> int:
    if "[ERROR]" in generated:
        return 1
    if "[fallback:" in generated:
        return 3
    if "不知道" in generated or "无法确定" in generated:
        return 3
    return max(1, min(5, round(1 + similarity * 4)))


def run_evaluation() -> dict[str, float]:
    random.seed(12)
    agent = EvalAgent()
    hallucinated = 0
    total_score = 0

    print(f"后端：{backend_name()}")
    print(f"评估问题数：{len(EVAL_SET)}")

    for index, item in enumerate(EVAL_SET, start=1):
        answer = agent.ask(item.question)
        is_hallucinated, similarity = hallucination_detect(answer, item.reference)
        score = user_score(answer, similarity)
        total_score += score
        hallucinated += int(is_hallucinated)

        mark = "疑似幻觉" if is_hallucinated else "参考相近"
        print(f"\n{index}. {item.question}")
        print(f"生成回答：{answer}")
        print(f"参考答案：{item.reference}")
        print(f"相似度：{similarity:.2f} | {mark} | 满意度：{score}/5")

    total = len(EVAL_SET)
    halluc_rate = hallucinated / total
    avg_score = total_score / total

    print("\n评估结果统计")
    print(f"总问题数：{total}")
    print(f"疑似幻觉数：{hallucinated}")
    print(f"疑似幻觉率：{halluc_rate:.2f}")
    print(f"平均满意度：{avg_score:.2f}/5")

    return {"hallucination_rate": halluc_rate, "avg_score": avg_score}


def main() -> None:
    run_evaluation()


if __name__ == "__main__":
    main()
