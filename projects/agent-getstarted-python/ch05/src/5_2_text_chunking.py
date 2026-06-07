# 【例5-2】
# text_chunking.py
#
# 本例演示RAG中的文档切片策略：按句子累积成chunk，并保留少量overlap。
#
# 运行方式：
# cd /Users/dustchen/workdir/dev_agents/projects/agent-getstarted-python
# python3 ch05/src/5_2_text_chunking.py

from rag_utils import split_text_by_sentences


document_text = """
人工智能是计算机科学的一个分支，旨在研究和开发模拟、延伸和扩展人类智能的理论、方法、技术及应用系统。
它是一门综合性很强的学科，涉及哲学、数学、经济学、神经科学、计算机科学等多个领域。
随着计算能力和数据规模的迅猛增长，人工智能技术在自然语言处理、计算机视觉、智能控制等方面取得了显著进展。
其中，大语言模型的兴起极大提升了机器处理自然语言的能力。
然而，为了使这些模型在问答、对话等场景下具备更强的事实性与可控性，往往需要引入检索增强机制（RAG）以结合外部知识源。
在此过程中，文档切片（chunking）策略成为RAG系统性能优化的重要一环。
"""

chunks = split_text_by_sentences(document_text, max_len=100, overlap=20)

for index, chunk in enumerate(chunks, start=1):
    print(f"Chunk {index} | len={len(chunk)}")
    print(chunk)
    print()
