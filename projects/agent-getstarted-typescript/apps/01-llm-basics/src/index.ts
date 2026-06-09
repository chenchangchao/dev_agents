/**
 * 01：LLM 基础调用。
 *
 * 运行：
 * bun run apps/01-llm-basics/src/index.ts
 * LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=qwen3:1.7b bun run apps/01-llm-basics/src/index.ts
 */

import { backendName, chatText } from "@agent/core";

export async function main(): Promise<void> {
  console.log(`后端：${backendName()}`);
  const answer = await chatText("用三句话解释什么是Agent。", {
    system: "你是一名中文技术讲师，回答要简洁。",
    temperature: 0.3,
    maxTokens: 220,
    fallbackLabel: "llm-basics"
  });
  console.log(answer);
}

if (import.meta.main) {
  await main();
}
