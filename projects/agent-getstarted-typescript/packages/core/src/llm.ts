import { boolEnv, env, selectedBackend } from "./env";
import type { ChatOptions, Message } from "./types";

interface OllamaChatResponse {
  message?: {
    content?: string;
  };
}

interface CloudChatResponse {
  choices?: Array<{
    message?: {
      content?: string;
    };
  }>;
}

function fallbackText(prompt: string, label: string, error: unknown): string {
  const reason = error instanceof Error ? error.name : "UnknownError";
  const shortPrompt = prompt.replace(/\s+/g, " ").slice(0, 180);
  return `[fallback:${label}] 当前模型后端不可用，已使用TypeScript教学回退结果。输入摘要：“${shortPrompt}”。建议继续验证Agent流程、工具调用、RAG上下文和服务接口。原始异常：${reason}`;
}

function cloudConfig(options: ChatOptions): { baseUrl: string; apiKey: string; model: string } {
  const deepseekKey = env("DEEPSEEK_API_KEY");
  const openaiKey = env("OPENAI_API_KEY");
  return {
    baseUrl: env("DEEPSEEK_BASE_URL") || env("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    apiKey: deepseekKey || openaiKey,
    model: options.model || env("DEEPSEEK_MODEL_ID") || env("OPENAI_MODEL_ID", "gpt-4o-mini")
  };
}

export async function chatMessages(messages: Message[], options: ChatOptions = {}): Promise<string> {
  const backend = options.backend ?? selectedBackend();
  const temperature = options.temperature ?? 0.7;
  const maxTokens = options.maxTokens;
  const label = options.fallbackLabel ?? "chat";

  try {
    if (backend === "ollama") {
      const baseUrl = env("OLLAMA_BASE_URL", "http://127.0.0.1:11434").replace(/\/$/, "");
      const response = await fetch(`${baseUrl}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model: options.model || env("OLLAMA_MODEL", "qwen3:1.7b"),
          messages,
          stream: false,
          think: false,
          options: {
            temperature,
            ...(maxTokens ? { num_predict: maxTokens } : {})
          }
        })
      });
      if (!response.ok) throw new Error(`Ollama HTTP ${response.status}`);
      const data = (await response.json()) as OllamaChatResponse;
      const content = String(data?.message?.content ?? "").trim();
      if (!content) throw new Error("Ollama returned empty content");
      return content;
    }

    const cloud = cloudConfig(options);
    if (!cloud.apiKey) throw new Error("Missing cloud API key");
    const response = await fetch(`${cloud.baseUrl.replace(/\/$/, "")}/chat/completions`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${cloud.apiKey}`
      },
      body: JSON.stringify({
        model: cloud.model,
        messages,
        temperature,
        ...(maxTokens ? { max_tokens: maxTokens } : {})
      })
    });
    if (!response.ok) throw new Error(`Cloud HTTP ${response.status}`);
    const data = (await response.json()) as CloudChatResponse;
    const content = String(data?.choices?.[0]?.message?.content ?? "").trim();
    if (!content) throw new Error("Cloud model returned empty content");
    return content;
  } catch (error) {
    if (!boolEnv("AGENT_TS_LLM_FALLBACK", true)) throw error;
    return fallbackText(messages.map((message) => `${message.role}: ${message.content}`).join("\n"), label, error);
  }
}

export async function chatText(prompt: string, options: ChatOptions = {}): Promise<string> {
  const messages: Message[] = [];
  if (options.system) messages.push({ role: "system", content: options.system });
  messages.push({ role: "user", content: prompt });
  return chatMessages(messages, options);
}
