/**
 * 05：多 Agent 协作。
 *
 * 运行：
 * bun run apps/05-multi-agent/src/index.ts
 * LOCAL_LLM_BACKEND=ollama OLLAMA_MODEL=qwen3:1.7b bun run apps/05-multi-agent/src/index.ts
 */

import { backendName, chatText } from "@agent/core";
import { WindowMemory } from "@agent/memory";

class RoleAgent {
  constructor(
    readonly name: string,
    private readonly system: string
  ) {}

  async run(input: string): Promise<string> {
    return chatText(input, {
      system: this.system,
      temperature: 0.3,
      maxTokens: 260,
      fallbackLabel: this.name
    });
  }
}

export async function main(): Promise<void> {
  console.log(`后端：${backendName()}`);
  const memory = new WindowMemory(8);
  const planner = new RoleAgent("planner", "你是任务规划Agent，请输出简洁步骤。");
  const researcher = new RoleAgent("researcher", "你是研究Agent，请补充关键事实与风险。");
  const writer = new RoleAgent("writer", "你是写作Agent，请整合上游结果生成清晰结论。");

  const task = "设计一个本地优先的Agent学习项目路线";
  const plan = await planner.run(task);
  memory.add("assistant", `planner: ${plan}`);
  const research = await researcher.run(`${task}\n\n已有规划：${plan}`);
  memory.add("assistant", `researcher: ${research}`);
  const final = await writer.run(`${task}\n\n协作上下文：\n${memory.text()}`);

  console.log("\n[Planner]\n", plan);
  console.log("\n[Researcher]\n", research);
  console.log("\n[Writer]\n", final);
}

if (import.meta.main) {
  await main();
}
