# 【例3-4】
# qwen3_colab_batch_inference.py
#
# Colab T4推荐安装：
# !pip uninstall -y torchvision torchao
# !pip install -U torch transformers accelerate
# 运行时如果刚卸载/升级过torch相关包，请重启Colab Runtime。

import time
from importlib import metadata

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"
TORCHAO_MIN_VERSION = (0, 16, 0)


def parse_version(version: str) -> tuple[int, ...]:
    parts = []
    for part in version.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if digits:
            parts.append(int(digits))
        else:
            break
    return tuple(parts)


def check_colab_environment():
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA GPU。请在Colab中启用 Runtime > Change runtime type > T4 GPU。")

    try:
        torchao_version = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        torchao_version = None

    if torchao_version and parse_version(torchao_version) < TORCHAO_MIN_VERSION:
        raise RuntimeError(
            "检测到 Colab 中的 torchao 版本过旧："
            f"{torchao_version}。本示例不需要 torchao，"
            "请先执行：!pip uninstall -y torchao，然后重启 Colab Runtime。"
        )

    print(">>> CUDA GPU:", torch.cuda.get_device_name(0))
    print(">>> CUDA Memory:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2), "GB")


def load_model_and_tokenizer():
    print(">>> 正在加载模型与Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16,
    )
    model.eval()
    print(">>> 模型加载完成")
    return model, tokenizer


def build_prompt(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "你是一个中文技术助教，请用准确、简洁、分点的方式回答。",
        },
        {
            "role": "user",
            "content": question,
        },
    ]


def run_inference(model, tokenizer, question: str, max_new_tokens: int = 192) -> dict:
    inputs = tokenizer.apply_chat_template(
        build_prompt(question),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    start = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start

    input_len = inputs["input_ids"].shape[-1]
    new_tokens = outputs[0][input_len:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    token_count = len(new_tokens)

    return {
        "question": question,
        "answer": answer,
        "seconds": elapsed,
        "new_tokens": token_count,
        "tokens_per_second": token_count / elapsed if elapsed > 0 else 0,
    }


def print_result(result: dict):
    print("\n" + "=" * 80)
    print("问题：", result["question"])
    print("-" * 80)
    print(result["answer"])
    print("-" * 80)
    print(
        "生成token数：{new_tokens} | 耗时：{seconds:.2f}s | 速度：{tokens_per_second:.2f} tokens/s".format(
            **result
        )
    )


def validate_outputs(results: list[dict]):
    print("\n" + "=" * 80)
    print("批量推理验证")
    print("=" * 80)

    valid_count = 0
    for result in results:
        answer = result["answer"]
        is_valid = bool(answer) and len(answer) >= 20 and "Traceback" not in answer
        valid_count += int(is_valid)
        status = "通过" if is_valid else "需检查"
        print(f"- {status} | {result['question'][:28]}... | {result['new_tokens']} tokens")

    avg_speed = sum(item["tokens_per_second"] for item in results) / len(results)
    print(f"\n有效输出：{valid_count}/{len(results)}")
    print(f"平均生成速度：{avg_speed:.2f} tokens/s")


def main():
    check_colab_environment()
    model, tokenizer = load_model_and_tokenizer()

    prompts = [
        "请简述Transformer模型的核心机制。",
        "什么是多模态大模型？其应用场景有哪些？",
        "如何通过LoRA对大语言模型进行微调？",
        "当前AI Agent的主要技术难点有哪些？",
        "请解释RAG和微调的区别，并说明适用场景。",
        "为什么大模型推理时显存占用会随着上下文长度增加？",
        "请用表格比较CPU、GPU和TPU在深度学习中的作用。",
        "在Colab T4上部署小模型时，应该如何避免OOM？",
        "请解释什么是KV Cache，它对推理速度有什么影响？",
        "如何评估一个中文情感分类微调模型的效果？",
    ]

    print(f">>> 开始批量推理，共 {len(prompts)} 条样本")
    results = []
    for prompt in prompts:
        result = run_inference(model, tokenizer, prompt)
        results.append(result)
        print_result(result)
        torch.cuda.empty_cache()

    validate_outputs(results)


if __name__ == "__main__":
    main()
