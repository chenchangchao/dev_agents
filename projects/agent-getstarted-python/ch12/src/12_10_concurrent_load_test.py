"""例12-10：多用户并发与轻量压测。

运行方式：
cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
python3 ch12/src/12_10_concurrent_load_test.py
LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=qwen3:1.7b python3 ch12/src/12_10_concurrent_load_test.py
"""

from __future__ import annotations

import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from ch12_runtime import ask_llm, backend_name, extract_city, extract_numbers, safe_multiply


def fake_weather(city: str) -> str:
    return f"{city}天气晴朗，26°C"


@dataclass
class LoadResult:
    user_id: int
    input_text: str
    output: str
    duration: float


class ConcurrentAgent:
    def run(self, text: str) -> str:
        city = extract_city(text)
        if city != "未知" and any(word in text for word in ["天气", "气温"]):
            return fake_weather(city)

        nums = extract_numbers(text)
        if len(nums) >= 2 and any(word in text for word in ["乘", "*", "计算"]):
            return safe_multiply(nums[0], nums[1])

        return ask_llm(
            text,
            system="你是一个面向压测示例的中文助手，回答要简洁。",
            temperature=0.2,
            max_tokens=160,
            label="load-test",
        )


def run_concurrent_test(concurrent_users: int = 12) -> list[LoadResult]:
    agent = ConcurrentAgent()
    random.seed(12)
    tasks = [
        "请查询北京的天气",
        "广州今天什么天气",
        "请帮我计算 12 * 13",
        "算一下9乘以7是多少",
        "帮我查查上海的气温",
        "用一句话解释A2A通信",
    ]

    def task_runner(user_id: int) -> LoadResult:
        text = random.choice(tasks)
        start = time.time()
        response = agent.run(text)
        return LoadResult(user_id, text, response, time.time() - start)

    print(f"后端：{backend_name()}")
    print(f"并发用户数：{concurrent_users}")

    start_time = time.time()
    results: list[LoadResult] = []
    with ThreadPoolExecutor(max_workers=concurrent_users) as executor:
        futures = [executor.submit(task_runner, index) for index in range(concurrent_users)]
        for future in as_completed(futures):
            results.append(future.result())

    total_time = time.time() - start_time
    success_count = sum(1 for item in results if "[ERROR]" not in item.output)
    avg_time = sum(item.duration for item in results) / len(results)
    qps = len(results) / total_time if total_time else 0

    print("\n测试结果统计")
    print(f"总请求数：{len(results)}")
    print(f"成功响应数：{success_count}")
    print(f"失败响应数：{len(results) - success_count}")
    print(f"平均响应时间：{avg_time:.2f}s")
    print(f"QPS：{qps:.2f}")

    for item in sorted(results, key=lambda row: row.user_id)[:5]:
        print(f"\n用户{item.user_id} 输入：{item.input_text}")
        print(f"耗时：{item.duration:.2f}s")
        print(f"响应：{item.output}")

    return results


def main() -> None:
    run_concurrent_test()


if __name__ == "__main__":
    main()
