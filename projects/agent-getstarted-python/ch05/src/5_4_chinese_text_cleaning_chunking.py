# 【例5-4】
# chinese_text_cleaning_chunking.py
#
# 本例演示中文文本清洗、断句、chunk生成，并用表格形式打印结果。
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# python3 ch05/src/5_4_chinese_text_cleaning_chunking.py

import re

from rag_utils import clean_text, split_text_by_sentences


raw_text = """
近年来，人工智能在各个领域取得了飞速发展，尤其是在自然语言处理方面。
大语言模型（如GPT、BERT等）的出现，使得计算机在理解和生成自然语言的能力上有了质的飞跃。
然而，模型的性能高度依赖于输入数据的质量，如何对原始文本进行高质量的清洗与切分，成为构建高效语义检索系统的基础。
以RAG为代表的检索增强生成方法，需要将文档划分为结构合理、语义完整的片段，才能在后续嵌入计算与召回阶段获得最优表现。
本文将探讨一种面向中文文本的清洗与句元切分方案。
"""


def split_sentences(text: str) -> list[str]:
    return [item for item in re.split(r"(?<=[。！？])", text) if item.strip()]


cleaned_text = clean_text(raw_text)
sentences = split_sentences(cleaned_text)
chunks = split_text_by_sentences(cleaned_text, max_len=80, overlap=0)

print("清洗后文本：")
print(cleaned_text)

print("\n句子：")
for index, sentence in enumerate(sentences, start=1):
    print(f"{index}. {sentence}")

print("\nChunks：")
for index, chunk in enumerate(chunks, start=1):
    print(f"{index}. len={len(chunk)} | {chunk}")
