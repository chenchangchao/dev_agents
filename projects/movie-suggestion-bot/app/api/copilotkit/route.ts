import {
  CopilotRuntime,
  OpenAIAdapter,
  copilotRuntimeNextJSAppRouterEndpoint,
} from "@copilotkit/runtime";
import { createOpenAI } from "@ai-sdk/openai";
import { NextRequest } from "next/server";
import OpenAI from "openai";

const model = process.env.DEEPSEEK_MODEL_NAME || "deepseek-chat";

const openai = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: process.env.DEEPSEEK_BASE_URL,
  timeout: 30_000,
});

const copilotKit = new CopilotRuntime();
const serviceAdapter = new OpenAIAdapter({
  openai,
  model,
  maxInputTokens: 4_000,
});

const deepseek = createOpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: process.env.DEEPSEEK_BASE_URL,
  name: "deepseek",
});

serviceAdapter.getLanguageModel = () => deepseek.chat(model);

export const POST = async (req: NextRequest) => {
  const { handleRequest } = copilotRuntimeNextJSAppRouterEndpoint({
    runtime: copilotKit,
    serviceAdapter,
    endpoint: "/api/copilotkit",
  });

  return handleRequest(req);
};
