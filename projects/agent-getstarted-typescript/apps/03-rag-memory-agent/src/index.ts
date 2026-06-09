/**
 * 03：RAG + 记忆 Agent。
 *
 * 运行：
 * bun run apps/03-rag-memory-agent/src/index.ts
 * LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=qwen3:1.7b bun run apps/03-rag-memory-agent/src/index.ts
 */

import { backendName, dataPath } from "@agent/core";
import { JsonlMemory, WindowMemory } from "@agent/memory";
import { SimpleRagAgent } from "@agent/rag";

const CORPUS = `
Agent 是一种能够感知上下文、调用工具并完成目标的智能程序。
RAG 是 Retrieval-Augmented Generation 的缩写，核心思想是在生成前先检索相关资料。
短期记忆通常保存在当前会话中，长期记忆可以写入文件、数据库或向量库。
Ollama 适合在 Mac 本地运行小型或中型开源模型，DeepSeek API 适合云端推理。
`;

export async function main(): Promise<void> {
  console.log(`后端：${backendName()}`);
  const memory = new WindowMemory(4);
  const persistentMemory = new JsonlMemory(dataPath("memory", "rag-agent.jsonl"));
  const rag = new SimpleRagAgent(CORPUS);

  const questions = ["RAG是什么？", "Agent 的记忆可以放在哪里？"];
  for (const question of questions) {
    memory.add("user", question);
    await persistentMemory.append({ role: "user", content: question });
    const answer = await rag.ask(`${memory.text()}\n当前问题：${question}`);
    memory.add("assistant", answer);
    await persistentMemory.append({ role: "assistant", content: answer });
    console.log(`\n问题：${question}`);
    console.log(`回答：${answer}`);
  }

  console.log(`\n持久化记忆：${dataPath("memory", "rag-agent.jsonl")}`);
}

if (import.meta.main) {
  await main();
}
