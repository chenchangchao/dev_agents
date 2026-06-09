import type { Tool, ToolCallResult } from "@agent/core";

const WEATHER: Record<string, string> = {
  北京: "北京天气晴，26°C",
  上海: "上海多云，25°C",
  广州: "广州小雨，28°C",
  深圳: "深圳晴，29°C"
};

export function extractCity(input: string): string | null {
  return Object.keys(WEATHER).find((city) => input.includes(city)) ?? null;
}

export const weatherTool: Tool = {
  name: "weather",
  description: "查询示例城市天气",
  canHandle(input: string): boolean {
    return /天气|气温|温度/.test(input);
  },
  call(input: string): ToolCallResult {
    const city = extractCity(input);
    if (!city) return { ok: false, name: this.name, output: "暂不支持该城市", error: "city_not_found" };
    return { ok: true, name: this.name, output: WEATHER[city] };
  }
};
