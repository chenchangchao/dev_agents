/**
 * 04：Bun 原生 OpenAI-compatible Server。
 *
 * 运行：
 * bun run apps/04-agent-server/src/index.ts
 * LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=qwen3:1.7b bun run apps/04-agent-server/src/index.ts
 *
 * 测试：
 * curl -X POST http://127.0.0.1:3020/v1/chat/completions \
 *   -H "Content-Type: application/json" \
 *   -d '{"messages":[{"role":"user","content":"你好，介绍一下你自己"}],"max_tokens":120}'
 */

import { env } from "@agent/core";
import { createOpenAiCompatibleServer } from "@agent/server";

export function main(): void {
  const port = Number(env("AGENT_TS_SERVER_PORT", "3020"));
  const server = createOpenAiCompatibleServer(port);
  console.log(`OpenAI-compatible API: http://127.0.0.1:${server.port}/v1/chat/completions`);
  console.log(`Health check: http://127.0.0.1:${server.port}/health`);
}

if (import.meta.main) {
  main();
}
