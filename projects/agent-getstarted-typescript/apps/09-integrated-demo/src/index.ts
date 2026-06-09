/**
 * 09：综合 Agent Demo
 *
 * 运行：
 * bun run apps/09-integrated-demo/src/index.ts
 * docker compose -f docker/docker-compose.redis.yml up -d
 * AGENT_TS_API_TOKEN=dev-token bun run apps/09-integrated-demo/src/index.ts
 *
 * 测试：
 * curl http://127.0.0.1:3090/health
 * curl -X POST http://127.0.0.1:3090/chat \
 *   -H "Authorization: Bearer dev-token" \
 *   -H "Content-Type: application/json" \
 *   -d '{"sessionId":"demo-1","message":"请解释一下什么是RAG"}'
 */

import { ToolCallingAgent, dataPath, env } from "@agent/core";
import { RedisMemory, SqliteMemory, WindowMemory } from "@agent/memory";
import { AgentRegistry, InMemoryMessageBus, RedisMessageBus, createA2AMessage } from "@agent/protocols";
import { SimpleRagAgent } from "@agent/rag";
import { createFastifyAgentServer } from "@agent/server";
import { calculatorTool, weatherTool } from "@agent/tools";

const CORPUS = `
RAG 是 Retrieval-Augmented Generation，核心思想是在生成前先检索资料。
BM25 适合关键词匹配，向量检索更擅长语义接近，混合检索通常更稳。
Agent 可以结合工具、记忆和协议来完成复杂任务。
SQLite 适合本地轻量持久化，Redis 适合高速缓存与消息分发。
Fastify 适合构建高性能 Node/Bun 风格 API 服务。
`;

class IntegratedAgentService {
  private readonly rag = new SimpleRagAgent(CORPUS);
  private readonly tools = new ToolCallingAgent([weatherTool, calculatorTool]);
  private readonly windowBySession = new Map<string, WindowMemory>();
  private readonly sqlite = new SqliteMemory(dataPath("memory", env("SQLITE_MEMORY_PATH", "integrated-memory.sqlite")));
  private readonly redisMemory = new RedisMemory();
  private readonly registry = new AgentRegistry();
  private readonly bus = new InMemoryMessageBus();
  private redisBus: RedisMessageBus | null = null;

  async init(): Promise<void> {
    this.registry.register({ id: "planner", name: "Planner Agent", capabilities: ["route"] });
    this.registry.register({ id: "retriever", name: "Retriever Agent", capabilities: ["retrieve"] });
    this.registry.register({ id: "writer", name: "Writer Agent", capabilities: ["respond"] });

    let redisBus: RedisMessageBus | null = null;
    try {
      redisBus = new RedisMessageBus();
      await redisBus.publish("agent.bootstrap", createA2AMessage({
        from: "integrated",
        to: "redis",
        type: "event",
        payload: { ok: true }
      }));
      this.redisBus = redisBus;
    } catch {
      if (redisBus) {
        await redisBus.close().catch(() => undefined);
      }
      this.redisBus = null;
    }
  }

  private memory(sessionId: string): WindowMemory {
    const existing = this.windowBySession.get(sessionId);
    if (existing) return existing;
    const memory = new WindowMemory(8);
    this.windowBySession.set(sessionId, memory);
    return memory;
  }

  async handleMessage(sessionId: string, message: string): Promise<{ answer: string; route: string; refs?: string[] }> {
    const memory = this.memory(sessionId);
    memory.add("user", message);
    this.sqlite.append(sessionId, { role: "user", content: message });
    await this.redisMemory.append(sessionId, { role: "user", content: message }).catch(() => undefined);

    let route = "llm";
    let answer = "";
    let refs: string[] | undefined;

    if (weatherTool.canHandle(message) || calculatorTool.canHandle(message)) {
      route = "tool";
      answer = (await this.tools.run(message)).output;
    } else if (/RAG|检索|记忆|Agent|SQLite|Redis|Fastify/i.test(message)) {
      route = "rag";
      refs = this.rag.retrieve(message, "hybrid");
      answer = await this.rag.ask(`${memory.text()}\n\n当前问题：${message}`, "hybrid");
    } else {
      route = "rag-lite";
      answer = await this.rag.ask(message, "bm25");
    }

    memory.add("assistant", answer);
    this.sqlite.append(sessionId, { role: "assistant", content: answer });
    await this.redisMemory.append(sessionId, { role: "assistant", content: answer }).catch(() => undefined);

    const event = createA2AMessage({
      from: "writer",
      to: "auditor",
      type: "event",
      payload: { sessionId, route, preview: answer.slice(0, 80) }
    });
    await this.bus.publish("agent.audit", event);
    if (this.redisBus) {
      await this.redisBus.publish("agent.audit", event).catch(() => undefined);
    }

    return { answer, route, refs };
  }

  readSession(sessionId: string): { sqlite: unknown; window: string } {
    return {
      sqlite: this.sqlite.read(sessionId, 20),
      window: this.memory(sessionId).text()
    };
  }

  listRegistry(): unknown[] {
    return this.registry.list();
  }

  async close(): Promise<void> {
    this.sqlite.close();
    await this.redisMemory.close().catch(() => undefined);
    if (this.redisBus) await this.redisBus.close().catch(() => undefined);
  }
}

export async function main(): Promise<void> {
  const app = createFastifyAgentServer();
  const service = new IntegratedAgentService();
  await service.init();

  app.get("/registry", async () => service.listRegistry());
  app.get("/memory/:sessionId", async (request) => {
    const { sessionId } = request.params as { sessionId: string };
    return service.readSession(sessionId);
  });
  app.post("/chat", async (request) => {
    const body = (request.body ?? {}) as { sessionId?: string; message?: string };
    const sessionId = body.sessionId ?? "default-session";
    const message = body.message?.trim() ?? "";
    return service.handleMessage(sessionId, message);
  });

  app.addHook("onClose", async () => {
    await service.close();
  });

  const port = Number(env("AGENT_TS_FASTIFY_PORT", "3090"));
  await app.listen({ host: "0.0.0.0", port });
  app.log.info({ port }, "integrated_demo_ready");
}

if (import.meta.main) {
  await main();
}
