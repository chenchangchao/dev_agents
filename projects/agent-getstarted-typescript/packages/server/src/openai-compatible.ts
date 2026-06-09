import { chatMessages, env, type Message } from "@agent/core";
import { openAiCompatibleEventStream } from "./stream";

interface ChatRequest {
  model?: string;
  messages?: Message[];
  temperature?: number;
  max_tokens?: number;
  stream?: boolean;
}

function json(data: unknown, status = 200): Response {
  return new Response(JSON.stringify(data), {
    status,
    headers: { "Content-Type": "application/json; charset=utf-8" }
  });
}

export async function handleChatCompletions(req: Request): Promise<Response> {
  if (req.method !== "POST") return json({ error: "method_not_allowed" }, 405);
  const body = (await req.json()) as ChatRequest;
  const messages = body.messages ?? [];
  const model = body.model ?? env("OLLAMA_MODEL", "local-or-cloud");
  const content = await chatMessages(messages, {
    model,
    temperature: body.temperature,
    maxTokens: body.max_tokens,
    fallbackLabel: "server"
  });
  if (body.stream) {
    return openAiCompatibleEventStream(content, model);
  }
  return json({
    id: `chatcmpl-${crypto.randomUUID()}`,
    object: "chat.completion",
    model,
    choices: [
      {
        index: 0,
        message: { role: "assistant", content },
        finish_reason: "stop"
      }
    ]
  });
}

export function createOpenAiCompatibleServer(port = Number(env("AGENT_TS_SERVER_PORT", "3020"))): ReturnType<typeof Bun.serve> {
  return Bun.serve({
    port,
    async fetch(req) {
      const url = new URL(req.url);
      if (url.pathname === "/health") return json({ ok: true });
      if (url.pathname === "/v1/chat/completions") return handleChatCompletions(req);
      return json({ error: "not_found" }, 404);
    }
  });
}
