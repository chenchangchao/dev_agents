import type { Tool, ToolCallResult } from "@agent/core";

export function safeCalculate(expression: string): number {
  if (!/^[\d\s+\-*/().]+$/.test(expression)) {
    throw new Error("表达式包含不允许的字符");
  }
  const fn = new Function(`"use strict"; return (${expression});`);
  const result = fn();
  if (typeof result !== "number" || !Number.isFinite(result)) {
    throw new Error("计算结果无效");
  }
  return result;
}

function extractExpression(input: string): string | null {
  const match = input.match(/[\d\s+\-*/().]+/);
  return match?.[0]?.trim() || null;
}

export const calculatorTool: Tool = {
  name: "calculator",
  description: "处理四则运算表达式",
  canHandle(input: string): boolean {
    return /计算|算一下|乘|加|\*|\/|\+|-/.test(input) && /\d/.test(input);
  },
  call(input: string): ToolCallResult {
    try {
      const normalized = input.replaceAll("乘以", "*").replaceAll("乘", "*").replaceAll("加", "+");
      const expression = extractExpression(normalized);
      if (!expression) throw new Error("没有识别到表达式");
      return { ok: true, name: this.name, output: String(safeCalculate(expression)) };
    } catch (error) {
      return { ok: false, name: this.name, output: "计算失败", error: error instanceof Error ? error.message : String(error) };
    }
  }
};
