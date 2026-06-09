/**
 * 07：消息总线 Agent。
 *
 * 优先尝试 Redis pub/sub，连接失败时回退到内存消息总线。
 *
 * 运行：
 * bun run apps/07-redis-agents/src/index.ts
 *
 * 可选Redis：
 * docker compose -f docker/docker-compose.redis.yml up -d
 */

import { InMemoryMessageBus, RedisMessageBus, createA2AMessage, type MessageBus } from "@agent/protocols";

export async function main(): Promise<void> {
  let bus: MessageBus = new InMemoryMessageBus();
  let mode = "memory";
  try {
    const redisBus = new RedisMessageBus();
    await redisBus.publish("agent.bootstrap", createA2AMessage({
      from: "system",
      to: "redis",
      type: "event",
      payload: { ping: true }
    }));
    bus = redisBus;
    mode = "redis";
  } catch {
    if (bus instanceof RedisMessageBus) {
      await bus.close().catch(() => undefined);
    }
    bus = new InMemoryMessageBus();
  }
  const received: string[] = [];

  bus.subscribe<{ event: string }>("agent.events", (message) => {
    received.push(`${message.from}->${message.to}:${message.payload.event}`);
    console.log("[subscriber]", received.at(-1));
  });

  await bus.publish(
    "agent.events",
    createA2AMessage({
      from: "scheduler",
      to: "worker",
      type: "event",
      payload: { event: "refresh-rag-index" }
    })
  );
  await bus.publish(
    "agent.events",
    createA2AMessage({
      from: "worker",
      to: "auditor",
      type: "event",
      payload: { event: "task-finished" }
    })
  );

  console.log(`总线模式：${mode}`);
  console.log(`消息数：${received.length}`);
  if (bus instanceof RedisMessageBus) {
    await bus.close().catch(() => undefined);
  }
}

if (import.meta.main) {
  await main();
}
