import "dotenv/config";
import OpenAI from "openai";

const apiKey = process.env.DEEPSEEK_API_KEY;
const baseURL = process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com";
const model = process.env.DEEPSEEK_MODEL_NAME || "deepseek-chat";

if (!apiKey) {
  console.error("DEEPSEEK_API_KEY is not configured");
  process.exit(1);
}

const client = new OpenAI({
  apiKey,
  baseURL,
  timeout: 30_000,
});

async function main() {
  console.log("Testing DeepSeek API connectivity...");
  console.log(`Base URL: ${baseURL}`);
  console.log(`Model: ${model}`);

  const response = await client.chat.completions.create({
    model,
    messages: [
      {
        role: "user",
        content:
          "Reply with exactly: DeepSeek connectivity OK",
      },
    ],
    max_tokens: 20,
    temperature: 0,
  });

  const content = response.choices[0]?.message?.content?.trim();

  if (!content) {
    throw new Error("DeepSeek response did not include message content");
  }

  console.log(`Response OK: ${content}`);
  console.log("DeepSeek API connectivity test passed.");
}

main().catch((error) => {
  console.error(
    "DeepSeek API connectivity test failed:",
    error instanceof Error ? error.message : error
  );
  process.exit(1);
});
