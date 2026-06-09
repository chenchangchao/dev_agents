import { afterEach, beforeEach, describe, expect, test } from "bun:test";
import { rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { ToolCallingAgent, chatText } from "@agent/core";
import { isLikelyHallucination, roughSimilarity } from "@agent/evals";
import { runLoadTest } from "@agent/evals";
import { SqliteMemory, WindowMemory } from "@agent/memory";
import { AgentRegistry, InMemoryMessageBus, RedisMessageBus, createA2AMessage, createMcpContext } from "@agent/protocols";
import { SimpleRagAgent, bm25Search, chunkText, hybridSearch, keywordSearch, vectorSearch } from "@agent/rag";
import { AsyncTaskQueue, FixedWindowRateLimiter, createFastifyAgentServer, handleChatCompletions, openAiCompatibleEventStream, textEventStream } from "@agent/server";
import { calculatorTool, safeCalculate, summarizeHtml, weatherTool } from "@agent/tools";

const oldEnv = { ...Bun.env };
const managedEnvKeys = [
  "LOCAL_LLM_BACKEND",
  "DEEPSEEK_API_KEY",
  "OPENAI_API_KEY",
  "AGENT_TS_LLM_FALLBACK",
  "AGENT_TS_API_TOKEN"
] as const;

beforeEach(() => {
  Bun.env.LOCAL_LLM_BACKEND = "";
  Bun.env.DEEPSEEK_API_KEY = "";
  Bun.env.OPENAI_API_KEY = "";
  Bun.env.AGENT_TS_LLM_FALLBACK = "1";
  Bun.env.AGENT_TS_API_TOKEN = "dev-token";
});

afterEach(() => {
  for (const key of managedEnvKeys) {
    Bun.env[key] = oldEnv[key] ?? "";
  }
});

describe("core llm", () => {
  test("falls back when cloud key is missing", async () => {
    const output = await chatText("测试fallback", { fallbackLabel: "unit" });
    expect(output).toContain("[fallback:unit]");
  });
});

describe("tools", () => {
  test("calculates safe expressions", () => {
    expect(safeCalculate("12 * 13")).toBe(156);
  });

  test("tool agent uses weather and calculator before llm", async () => {
    const agent = new ToolCallingAgent([weatherTool, calculatorTool]);
    const weather = await agent.run("北京天气怎么样");
    const calc = await agent.run("计算 7 * 8");
    expect(weather.tool?.name).toBe("weather");
    expect(calc.tool?.name).toBe("calculator");
    expect(calc.output).toBe("56");
  });

  test("summarizes html title and description", () => {
    const summary = summarizeHtml("<title>测试页</title><meta name=\"description\" content=\"页面描述\"><main>正文</main>");
    expect(summary.title).toBe("测试页");
    expect(summary.description).toBe("页面描述");
    expect(summary.text).toContain("正文");
  });
});

describe("rag and memory", () => {
  test("chunks and retrieves relevant text", () => {
    const chunks = chunkText("Agent 可以调用工具。RAG 会先检索资料再生成回答。", 12, 2);
    const results = keywordSearch("RAG 检索", chunks, 2);
    expect(results.length).toBeGreaterThan(0);
  });

  test("rag agent exposes retrieve", () => {
    const rag = new SimpleRagAgent("RAG 是检索增强生成。Agent 可以使用工具。");
    expect(rag.retrieve("什么是RAG").join("")).toContain("RAG");
  });

  test("bm25, vector and hybrid search all return ranked chunks", () => {
    const chunks = ["RAG 会先检索资料再生成回答", "天气工具查询北京气温", "SQLite 用于本地存储"];
    expect(bm25Search("检索 资料", chunks, 2).length).toBeGreaterThan(0);
    expect(vectorSearch("本地 存储", chunks, 2).length).toBeGreaterThan(0);
    expect(hybridSearch("北京 气温", chunks, 2).length).toBeGreaterThan(0);
  });

  test("window memory clips old messages", () => {
    const memory = new WindowMemory(2);
    memory.add("user", "a");
    memory.add("assistant", "b");
    memory.add("user", "c");
    expect(memory.all().map((item) => item.content)).toEqual(["b", "c"]);
  });
});

describe("protocols and server", () => {
  test("creates A2A and MCP messages", () => {
    const message = createA2AMessage({ from: "planner", to: "worker", type: "request", payload: { task: "run" } });
    const context = createMcpContext([{ role: "user", content: "hello" }], { app: "test" });
    expect(message.id).toBeTruthy();
    expect(context.metadata.app).toBe("test");
  });

  test("registry and memory bus route messages", async () => {
    const registry = new AgentRegistry();
    const bus = new InMemoryMessageBus();
    registry.register({ id: "writer", name: "Writer", capabilities: ["write"] });

    const received: string[] = [];
    bus.subscribe<{ text: string }>("writer", (message) => {
      received.push(message.payload.text);
    });
    await bus.publish("writer", createA2AMessage({ from: "a", to: "writer", type: "request", payload: { text: "hello" } }));

    expect(registry.findByCapability("write")[0].id).toBe("writer");
    expect(received).toEqual(["hello"]);
  });

  test("redis message bus can be constructed", () => {
    const bus = new RedisMessageBus({ url: "redis://127.0.0.1:6379" });
    expect(bus).toBeTruthy();
  });

  test("rate limiter blocks after limit", () => {
    const limiter = new FixedWindowRateLimiter(2, 1000);
    expect(limiter.check("u").allowed).toBe(true);
    expect(limiter.check("u").allowed).toBe(true);
    expect(limiter.check("u").allowed).toBe(false);
  });

  test("async task queue completes tasks", async () => {
    const queue = new AsyncTaskQueue<string>(async (input) => input.toUpperCase());
    const task = queue.enqueue("ok");
    await new Promise((resolve) => setTimeout(resolve, 5));
    expect(queue.get(task.id)?.status).toBe("done");
    expect(queue.get(task.id)?.result).toBe("OK");
  });

  test("event stream emits chunks", async () => {
    const response = textEventStream("hello", 2);
    const text = await response.text();
    expect(text).toContain("data:");
    expect(text).toContain("[DONE]");
  });

  test("openai compatible event stream emits chunk envelope", async () => {
    const response = openAiCompatibleEventStream("hello", "demo-model", 2);
    const text = await response.text();
    expect(text).toContain("chat.completion.chunk");
    expect(text).toContain("\"content\":\"he\"");
    expect(text).toContain("[DONE]");
  });

  test("openai compatible handler returns choices", async () => {
    const req = new Request("http://local/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }] })
    });
    const response = await handleChatCompletions(req);
    const data = (await response.json()) as { choices: Array<{ message: { content: string } }> };
    expect(response.status).toBe(200);
    expect(data.choices[0].message.content).toContain("[fallback:server]");
  });

  test("openai compatible handler supports stream", async () => {
    const req = new Request("http://local/v1/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: [{ role: "user", content: "hello" }], stream: true })
    });
    const response = await handleChatCompletions(req);
    expect(response.headers.get("content-type")).toContain("text/event-stream");
    const text = await response.text();
    expect(text).toContain("chat.completion.chunk");
    expect(text).toContain("[DONE]");
  });

  test("fastify server enforces auth and returns health", async () => {
    const app = createFastifyAgentServer();
    const health = await app.inject({ method: "GET", url: "/health" });
    const unauthorized = await app.inject({ method: "POST", url: "/v1/chat/completions", payload: { messages: [] } });
    const authorized = await app.inject({
      method: "POST",
      url: "/v1/chat/completions",
      headers: { authorization: "Bearer dev-token" },
      payload: { messages: [{ role: "user", content: "hello" }] }
    });
    expect(health.statusCode).toBe(200);
    expect(unauthorized.statusCode).toBe(401);
    expect(authorized.statusCode).toBe(200);
    await app.close();
  });
});

describe("evals", () => {
  test("rough hallucination heuristic works", () => {
    expect(roughSimilarity("蓝鲸是最大的哺乳动物", "最大的哺乳动物是蓝鲸")).toBeGreaterThan(0);
    expect(isLikelyHallucination("火星是奶酪做的", "地球上最大的哺乳动物是蓝鲸")).toBe(true);
  });

  test("load test runner records success", async () => {
    const result = await runLoadTest(5, 2, async (index) => `ok-${index}`);
    expect(result.total).toBe(5);
    expect(result.success).toBe(5);
    expect(result.samples.length).toBeGreaterThan(0);
  });
});

describe("persistent memory", () => {
  test("sqlite memory persists messages", () => {
    const file = join(tmpdir(), `agent-ts-${crypto.randomUUID()}.sqlite`);
    const memory = new SqliteMemory(file);
    memory.append("s1", { role: "user", content: "hello" });
    memory.append("s1", { role: "assistant", content: "world" });
    const rows = memory.read("s1", 10);
    expect(rows.map((row) => row.content)).toEqual(["hello", "world"]);
    memory.close();
    rmSync(file, { force: true });
  });
});
