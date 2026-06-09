import { chatText } from "./llm";
import type { AgentRunResult, Tool } from "./types";

export class ToolCallingAgent {
  constructor(
    private readonly tools: Tool[],
    private readonly system = "你是一个可靠的中文Agent。优先使用确定性工具，工具无法处理时再调用模型。"
  ) {}

  async run(input: string): Promise<AgentRunResult> {
    for (const tool of this.tools) {
      if (!tool.canHandle(input)) continue;
      const result = await tool.call(input);
      if (result.ok) return { output: result.output, tool: result };
      return { output: result.error ?? result.output, tool: result };
    }

    const output = await chatText(input, {
      system: this.system,
      maxTokens: 300,
      fallbackLabel: "tool-agent"
    });
    return { output, usedFallback: output.startsWith("[fallback:") };
  }
}
