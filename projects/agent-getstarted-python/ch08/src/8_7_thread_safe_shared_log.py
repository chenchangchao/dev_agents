# 【例8-7】状态同步与锁控制
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch08/src/8_7_thread_safe_shared_log.py
# 或使用云端DeepSeek/OpenAI兼容API：
# python3 ch08/src/8_7_thread_safe_shared_log.py

import threading
import time

from ch08_runtime import ask_llm, backend_name


class SharedLog:
    def __init__(self):
        self.logs: list[str] = []
        self.lock = threading.Lock()

    def write_log(self, agent_name: str, message: str) -> None:
        with self.lock:
            entry = f"[{agent_name}]：{message}"
            print(f"→ 写入日志：{entry}")
            self.logs.append(entry)
            time.sleep(0.1)

    def read_logs(self) -> str:
        with self.lock:
            return "\n".join(self.logs)


class LoggingAgent(threading.Thread):
    def __init__(self, name: str, log: SharedLog, query: str):
        super().__init__()
        self.name = name
        self.shared_log = log
        self.query = query
        self.response = ""

    def run(self) -> None:
        print(f"→ {self.name}开始处理任务")
        self.response = ask_llm(
            self.query,
            system="你是并发子Agent，请给出简短分析。",
            max_tokens=220,
            label=self.name,
        )
        self.shared_log.write_log(self.name, self.response)
        print(f"→ {self.name}完成写入")


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    print("=== 启动状态同步与锁控制模拟系统 ===")
    shared_log = SharedLog()
    queries = [
        "请分析中国当前的宏观经济政策",
        "请概述当前全球通货膨胀形势",
        "请简要预测人工智能在2025年的发展趋势",
    ]
    agents = [LoggingAgent(f"Agent_{index + 1}", shared_log, query) for index, query in enumerate(queries)]
    for agent in agents:
        agent.start()
    for agent in agents:
        agent.join()

    print("\n→ 所有Agent写入完成，准备总结：")
    print(shared_log.read_logs())

    summary_prompt = "以下是三个Agent的分析日志，请总结关键内容：\n" + shared_log.read_logs()
    final_result = ask_llm(summary_prompt, system="你是共享日志总结助手。", max_tokens=420, label="shared_log_summary")
    print("\n--- 输出总结 ---")
    print(final_result)


if __name__ == "__main__":
    main()
