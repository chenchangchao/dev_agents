/**
 * 02：工具调用 Agent。
 *
 * 运行：
 * bun run apps/02-agent-tools/src/index.ts
 * LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=qwen3:1.7b bun run apps/02-agent-tools/src/index.ts
 */

import { ToolCallingAgent, backendName } from "@agent/core";
import { calculatorTool, weatherTool } from "@agent/tools";
import { evaluateToolAccuracy } from "@agent/evals";

export async function main(): Promise<void> {
  console.log(`后端：${backendName()}`);
  const agent = new ToolCallingAgent([weatherTool, calculatorTool]);
  const inputs = [
    "请查一下北京天气",
    "帮我计算 12 * 13",
    "用一句话解释RAG是什么"
  ];

  for (const input of inputs) {
    const result = await agent.run(input);
    console.log(`\n用户：${input}`);
    console.log(`输出：${result.output}`);
    if (result.tool) console.log(`工具：${result.tool.name}`);
  }

  const evalResult = await evaluateToolAccuracy([weatherTool, calculatorTool], [
    { input: "上海气温多少", expectedTool: "weather" },
    { input: "计算 7 * 8", expectedTool: "calculator" }
  ]);
  console.log(`\n工具路由正确率：${evalResult.correct}/${evalResult.total} = ${evalResult.accuracy.toFixed(2)}`);
}

if (import.meta.main) {
  await main();
}
