export function env(name: string, fallback = ""): string {
  return Bun.env[name]?.trim() || fallback;
}

export function boolEnv(name: string, fallback = true): boolean {
  const value = Bun.env[name]?.trim();
  if (value === undefined || value === "") return fallback;
  return !["0", "false", "no", "off"].includes(value.toLowerCase());
}

export function selectedBackend(): "ollama" | "cloud" {
  return env("LOCAL_LLM_BACKEND").toLowerCase() === "ollama" ? "ollama" : "cloud";
}

export function backendName(): string {
  if (selectedBackend() === "ollama") {
    return `ollama:${env("OLLAMA_MODEL", "qwen3:1.7b")}`;
  }
  return `cloud:${env("DEEPSEEK_MODEL_ID") || env("OPENAI_MODEL_ID", "gpt-4o-mini")}`;
}
