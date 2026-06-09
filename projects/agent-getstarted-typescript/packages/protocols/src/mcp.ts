import type { Message } from "@agent/core";

export interface McpContext {
  sessionId: string;
  messages: Message[];
  metadata: Record<string, string>;
}

export function createMcpContext(messages: Message[], metadata: Record<string, string> = {}): McpContext {
  return {
    sessionId: crypto.randomUUID(),
    messages,
    metadata
  };
}
