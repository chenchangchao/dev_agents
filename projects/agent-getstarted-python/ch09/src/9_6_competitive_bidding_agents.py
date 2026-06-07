# 【例9-6】竞标式 Agent 任务分配
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch09/src/9_6_competitive_bidding_agents.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch09/src/9_6_competitive_bidding_agents.py

import random
import time
import uuid

from ch09_runtime import PromptEntry, ask_llm, backend_name


class BidMessage:
    def __init__(self, sender: str, task: str, proposal: str, score: float):
        self.id = str(uuid.uuid4())
        self.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self.sender = sender
        self.task = task
        self.proposal = proposal
        self.score = score

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "sender": self.sender,
            "task": self.task,
            "proposal": self.proposal,
            "score": self.score,
        }


class CompetitiveAgent:
    def __init__(self, name: str, expertise: str, capability_score: float):
        self.name = name
        self.expertise = expertise
        self.capability_score = capability_score

    def bid_for_task(self, task_desc: str) -> BidMessage:
        prompt = [
            PromptEntry(role="system", content=f"你是 {self.expertise} 领域的Agent，请生成任务解决策略。"),
            PromptEntry(role="user", content=f"当前任务为：{task_desc}，请提供你的解决方案建议。"),
        ]
        proposal = ask_llm(prompt, max_tokens=360, label=self.name)
        return BidMessage(self.name, task_desc, proposal, self.evaluate_score(proposal))

    def evaluate_score(self, content: str) -> float:
        return round(self.capability_score + random.uniform(-0.3, 0.3), 2)


class TaskOrchestrator:
    def __init__(self):
        self.agents: list[CompetitiveAgent] = []

    def register(self, agent: CompetitiveAgent) -> None:
        self.agents.append(agent)

    def dispatch_task(self, task_desc: str) -> str:
        print(f"\n【主控Agent广播任务】：{task_desc}")
        bids = []
        for agent in self.agents:
            bid = agent.bid_for_task(task_desc)
            print(f"→ {bid.sender} 报价，能力评分：{bid.score}")
            bids.append(bid)

        best_bid = max(bids, key=lambda item: item.score)
        print(f"\n【中标Agent】：{best_bid.sender}")
        print("中标方案：", best_bid.proposal)
        return best_bid.sender


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    random.seed(9)
    orchestrator = TaskOrchestrator()
    orchestrator.register(CompetitiveAgent("Macro-Agent", "宏观经济分析", 8.7))
    orchestrator.register(CompetitiveAgent("Data-Agent", "数据挖掘与预测", 8.4))
    orchestrator.register(CompetitiveAgent("Sentiment-Agent", "财经舆情解读", 8.6))
    orchestrator.dispatch_task("请分析人民币汇率波动对外贸企业的影响，并提出缓解建议。")


if __name__ == "__main__":
    main()
