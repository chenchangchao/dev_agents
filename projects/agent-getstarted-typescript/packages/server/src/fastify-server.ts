import Fastify, { type FastifyInstance, type FastifyReply, type FastifyRequest } from "fastify";
import { chatMessages, env, type Message } from "@agent/core";
import { openAiCompatibleEventStream } from "./stream";

interface ChatBody {
  model?: string;
  messages?: Message[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
}

function requireAuth(request: FastifyRequest, _reply: FastifyReply): void {
  const token = env("AGENT_TS_API_TOKEN", "dev-token");
  const auth = request.headers.authorization ?? "";
  const expected = `Bearer ${token}`;
  if (auth !== expected) {
    const error = new Error("unauthorized") as Error & { statusCode?: number };
    error.statusCode = 401;
    throw error;
  }
}

export function createFastifyAgentServer(): FastifyInstance {
  const app = Fastify({
    logger: true
  });

  app.addHook("onRequest", async (request, reply) => {
    if (request.url === "/health") return;
    requireAuth(request, reply);
  });

  app.addHook("onResponse", async (request, reply) => {
    app.log.info({
      method: request.method,
      url: request.url,
      statusCode: reply.statusCode,
      requestId: request.id
    }, "request_complete");
  });

  app.get("/health", async () => ({ ok: true, server: "fastify" }));

  app.post("/v1/chat/completions", async (request, reply) => {
    const body = (request.body ?? {}) as ChatBody;
    const messages = body.messages ?? [];
    const model = body.model ?? env("OLLAMA_MODEL", "local-or-cloud");
    const content = await chatMessages(messages, {
      model,
      temperature: body.temperature,
      maxTokens: body.max_tokens,
      fallbackLabel: "fastify-server"
    });
    if (body.stream) {
      reply.header("content-type", "text/event-stream; charset=utf-8");
      return reply.send(openAiCompatibleEventStream(content, model).body);
    }
    return {
      id: `chatcmpl-${crypto.randomUUID()}`,
      object: "chat.completion",
      model,
      choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: "stop" }]
    };
  });

  return app;
}
