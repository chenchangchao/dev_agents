# 【例4-4】
# 4_4_safe_calculator_agent.py
#
# 本例演示安全工具执行：
# 1. 只允许安全四则运算表达式
# 2. 使用signal限制执行时间
# 3. 非计算请求交给LLM做简短说明
#
# 本地Ollama运行：
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch04/src/4_4_safe_calculator_agent.py
#
# 云端DeepSeek/OpenAI兼容API运行：
# python3 ch04/src/4_4_safe_calculator_agent.py

import re
import signal

from llm_backend import ask_text, backend_name


# 限时执行器定义
class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("执行超时")


signal.signal(signal.SIGALRM, timeout_handler)


# 安全计算表达式的沙箱函数
def safe_eval(expression: str) -> str:
    signal.alarm(3)  # 限制最多3秒
    try:
        allowed_chars = "0123456789+-*/(). "
        if not all(c in allowed_chars for c in expression):
            return "非法表达式，包含禁止字符"
        result = eval(expression, {"__builtins__": {}})
        return f"结果为：{result}"
    except TimeoutException:
        return "执行超时"
    except Exception as e:
        return f"执行失败：{str(e)}"
    finally:
        signal.alarm(0)


def run_agent(user_input: str) -> str:
    expression_match = re.search(r"[\d+\-*/(). ]+", user_input)
    if expression_match and "计算" in user_input:
        return safe_eval(expression_match.group(0).strip())

    return ask_text(
        "请判断用户是否在请求数学表达式计算；如果不是，请简短说明只能处理安全四则运算。\n"
        f"用户输入：{user_input}",
        system="你是一个安全计算助手。回答不超过80字，不要输出思考过程。",
        temperature=0,
        max_tokens=200,
    )


def main():
    print(f"LLM后端：{backend_name()}\n")

    # 多轮调用验证沙箱效果
    print("用户输入：计算表达式 3*(5+2)")
    res1 = run_agent("计算表达式 3*(5+2)")
    print("模型输出：", res1)

    print("\n用户输入：执行表达式 import os; os.system('rm -rf /')")
    res2 = run_agent("执行表达式 import os; os.system('rm -rf /')")
    print("模型输出：", res2)


if __name__ == "__main__":
    main()
