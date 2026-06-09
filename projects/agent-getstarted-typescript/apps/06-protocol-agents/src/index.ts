/**
 * 06：MCP/A2A 协议化 Agent。
 *
 * 运行：
 * bun run apps/06-protocol-agents/src/index.ts
 */

import { chatText } from "@agent/core";
import { AgentRegistry, InMemoryMessageBus, createA2AMessage, createMcpContext } from "@agent/protocols";

export async function main(): Promise<void> {
  const registry = new AgentRegistry();
  const bus = new InMemoryMessageBus();

  registry.register({ id: "researcher", name: "Research Agent", capabilities: ["research"] });
  registry.register({ id: "writer", name: "Writer Agent", capabilities: ["write"] });

  bus.subscribe<{ prompt: string }>("agent.writer", async (message) => {
    const context = createMcpContext([{ role: "user", content: message.payload.prompt }], {
      from: message.from,
      to: message.to
    });
    const output = await chatText(`请根据MCP上下文生成回复：${JSON.stringify(context)}`, {
      maxTokens: 200,
      fallbackLabel: "protocol-writer"
    });
    console.log("[writer收到A2A消息]", message);
    console.log("[writer输出]", output);
  });

  const writer = registry.findByCapability("write")[0];
  const message = createA2AMessage({
    from: "researcher",
    to: writer.id,
    type: "request",
    payload: { prompt: "把RAG解释成一句适合初学者理解的话。" }
  });

  console.log("注册Agent：", registry.list());
  await bus.publish("agent.writer", message);
}

if (import.meta.main) {
  await main();
}
