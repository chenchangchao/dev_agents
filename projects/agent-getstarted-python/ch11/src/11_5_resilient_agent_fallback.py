# 【例11-5】智能体容错与备用后端
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch11/src/11_5_resilient_agent_fallback.py

import logging
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from ch11_runtime import ask_llm, backend_name, data_path

LOG_FILE = data_path("logs", "agent_error.log")
logging.basicConfig(filename=LOG_FILE, level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s")


class ResilientAgent:
    def __init__(self, primary: str = "primary", backup: str = "backup", max_retries: int = 2, timeout: int = 15):
        self.model_labels = [primary, backup]
        self.max_retries = max_retries
        self.timeout = timeout
        self.session_id = str(uuid.uuid4())
        self.history: list[tuple[str, str]] = []

    def _call_with_timeout(self, input_text: str, model_label: str) -> str:
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._safe_run_chain, input_text, model_label)
            return future.result(timeout=self.timeout)

    def _safe_run_chain(self, input_text: str, model_label: str) -> str:
        prompt = f"历史对话：{self.history}\n用户：{input_text}"
        return ask_llm(
            prompt,
            system=f"你是容错Agent当前尝试的模型配置：{model_label}。",
            temperature=0.4,
            max_tokens=500,
            label=model_label,
        )

    def _log_failure(self, error_msg: str) -> None:
        logging.error(f"Session {self.session_id} Failed:\n{error_msg}")

    def ask(self, user_input: str) -> str:
        for model_label in self.model_labels:
            for retry in range(self.max_retries):
                try:
                    print(f"[INFO] 尝试使用模型配置：{model_label}（第{retry + 1}次）")
                    response = self._call_with_timeout(user_input, model_label)
                    self.history.append((user_input, response))
                    return response
                except Exception as exc:
                    self._log_failure(traceback.format_exc())
                    print(f"[WARN] 模型{model_label}调用失败，错误：{exc}")

        fallback_msg = "很抱歉，我暂时无法回答您的问题，请稍后再试。"
        self.history.append((user_input, fallback_msg))
        return fallback_msg


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print(f"错误日志：{LOG_FILE}")
    agent = ResilientAgent(max_retries=2, timeout=10)
    for question in ["请解释注意力机制的原理", "它与Transformer的关系呢？", "多头注意力怎么提升模型性能？"]:
        print(f"\n[用户] {question}")
        print("[智能体回复]", agent.ask(question))


if __name__ == "__main__":
    main()
