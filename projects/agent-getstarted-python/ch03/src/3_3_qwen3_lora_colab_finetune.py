# 【例3-3】
# qwen3_lora_colab_finetune.py
#
# Colab T4推荐安装：
# !pip uninstall -y torchvision torchao
# !pip install -U torch transformers accelerate datasets peft
# 运行时如果刚卸载/升级过torch相关包，请重启Colab Runtime。

from importlib import metadata

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

MODEL_ID = "Qwen/Qwen3-1.7B"
OUTPUT_DIR = "./output/qwen3_lora_sentiment"
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


def check_colab_optional_dependencies():
    try:
        torchao_version = metadata.version("torchao")
    except metadata.PackageNotFoundError:
        return

    if parse_version(torchao_version) < TORCHAO_MIN_VERSION:
        raise RuntimeError(
            "检测到 Colab 中的 torchao 版本过旧："
            f"{torchao_version}。这个 LoRA 示例不需要 torchao，"
            "请先执行：!pip uninstall -y torchao，然后重启 Colab Runtime。"
        )


def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        device_map="auto",
        dtype=torch.float16,
    )
    model.config.use_cache = False
    return model, tokenizer


def build_dataset():
    data = [
        {"text": "客服态度很好，处理问题及时", "label": "正面"},
        {"text": "屏幕亮度太低，看起来很费眼", "label": "负面"},
        {"text": "物流太快了，包装也非常结实", "label": "正面"},
        {"text": "使用几天就死机了，非常糟糕", "label": "负面"},
        {"text": "价格优惠，功能也很全面", "label": "正面"},
        {"text": "音质不清晰，续航时间短", "label": "负面"},
    ]
    return Dataset.from_list(data)


def format_training_text(sample, tokenizer):
    messages = [
        {
            "role": "system",
            "content": "你是一个中文情感分类助手，只回答“正面”或“负面”。",
        },
        {
            "role": "user",
            "content": f"请判断以下句子的情感倾向：{sample['text']}",
        },
        {
            "role": "assistant",
            "content": sample["label"],
        },
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )


def tokenize_dataset(dataset, tokenizer):
    def preprocess(sample):
        text = format_training_text(sample, tokenizer)
        tokenized = tokenizer(
            text,
            truncation=True,
            max_length=256,
            padding=False,
        )
        tokenized["labels"] = tokenized["input_ids"].copy()
        return tokenized

    return dataset.map(preprocess, remove_columns=dataset.column_names)


def build_lora_model(base_model):
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()
    return model


def train():
    model, tokenizer = load_model_and_tokenizer()
    model = build_lora_model(model)

    dataset = build_dataset()
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    train_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        logging_steps=1,
        save_strategy="epoch",
        learning_rate=2e-4,
        fp16=True,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f">>> LoRA权重已保存到：{OUTPUT_DIR}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError("未检测到CUDA GPU。请在Colab中启用 Runtime > Change runtime type > T4 GPU。")
    check_colab_optional_dependencies()
    print(">>> CUDA GPU:", torch.cuda.get_device_name(0))
    train()
