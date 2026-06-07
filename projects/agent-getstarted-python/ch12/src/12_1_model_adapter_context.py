# 【例12-1】多模型适配器与统一上下文
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=gemma4:e2b-mlx python3 ch12/src/12_1_model_adapter_context.py

import traceback
from typing import Any

from ch12_runtime import ConversationContext, ask_with_context, backend_name


class ModelAdapter:
    def __init__(self, model_type: str = "default"):
        self.model_type = model_type

    def call(self, context: ConversationContext) -> dict[str, Any]:
        try:
            response = ask_with_context(context.as_dict(), temperature=0.7, max_tokens=512, label=self.model_type)
            return {"success": True, "response": response}
        except Exception as exc:
            return {"success": False, "error": str(exc), "traceback": traceback.format_exc()}


def main() -> None:
    print(f"LLM后端：{backend_name()}")
    context = ConversationContext()
    context.add_message("system", "你是一个专业的AI助手")
    context.add_message("user", "请简要解释什么是注意力机制")

    print("=== 示例：使用主模型进行对话 ===")
    result = ModelAdapter("primary").call(context)
    print("[模型回复]" if result["success"] else "[模型失败]", result.get("response") or result.get("error"))
    if result["success"]:
        context.add_message("assistant", result["response"])

    print("\n=== 继续对话 ===")
    context.add_message("user", "它和多头注意力有什么关系？")
    result2 = ModelAdapter("backup").call(context)
    print("[模型回复]" if result2["success"] else "[模型失败]", result2.get("response") or result2.get("error"))


if __name__ == "__main__":
    main()
