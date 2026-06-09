/**
 * 08：Agent 评估与压测。
 *
 * 运行：
 * bun run apps/08-agent-evals/src/index.ts
 */

import { ToolCallingAgent } from "@agent/core";
import { evaluateToolAccuracy, isLikelyHallucination, roughSimilarity, runLoadTest } from "@agent/evals";
import { calculatorTool, weatherTool } from "@agent/tools";

export async function main(): Promise<void> {
  const tools = [weatherTool, calculatorTool];
  const accuracy = await evaluateToolAccuracy(tools, [
    { input: "北京天气怎么样", expectedTool: "weather" },
    { input: "计算 9 * 7", expectedTool: "calculator" },
    { input: "上海气温", expectedTool: "weather" }
  ]);
  console.log(`工具正确率：${accuracy.correct}/${accuracy.total} = ${accuracy.accuracy.toFixed(2)}`);

  const answer = "蓝鲸是地球上最大的哺乳动物";
  const reference = "地球上最大的哺乳动物是蓝鲸";
  console.log(`相似度：${roughSimilarity(answer, reference).toFixed(2)}`);
  console.log(`疑似幻觉：${isLikelyHallucination("火星由奶酪构成", reference)}`);

  const agent = new ToolCallingAgent(tools);
  const load = await runLoadTest(12, 4, async (index) => {
    const input = index % 2 === 0 ? "北京天气怎么样" : "计算 12 * 13";
    const result = await agent.run(input);
    return result.output;
  });
  console.log(`压测：total=${load.total}, success=${load.success}, avgMs=${load.avgMs.toFixed(2)}, qps=${load.qps.toFixed(2)}`);
}

if (import.meta.main) {
  await main();
}
