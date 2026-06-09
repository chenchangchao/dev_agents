export type Role = "system" | "user" | "assistant" | "tool";

export interface Message {
  role: Role;
  content: string;
  name?: string;
}

export interface ChatOptions {
  system?: string;
  model?: string;
  temperature?: number;
  maxTokens?: number;
  backend?: "ollama" | "cloud";
  fallbackLabel?: string;
}

export interface ToolCallResult {
  ok: boolean;
  name: string;
  output: string;
  error?: string;
}

export interface Tool {
  name: string;
  description: string;
  canHandle(input: string): boolean;
  call(input: string): Promise<ToolCallResult> | ToolCallResult;
}

export interface AgentRunResult {
  output: string;
  tool?: ToolCallResult;
  usedFallback?: boolean;
}
