import type { Tool } from "@agent/core";

export interface ToolEvalCase {
  input: string;
  expectedTool: string;
}

export async function evaluateToolAccuracy(tools: Tool[], cases: ToolEvalCase[]): Promise<{ correct: number; total: number; accuracy: number }> {
  let correct = 0;
  for (const item of cases) {
    const tool = tools.find((candidate) => candidate.canHandle(item.input));
    if (tool?.name === item.expectedTool) correct += 1;
  }
  return {
    correct,
    total: cases.length,
    accuracy: cases.length ? correct / cases.length : 0
  };
}
