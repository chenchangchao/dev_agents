import { fetchMovies } from "@/apis/movies";
import OpenAI from "openai";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type ChatRequest = {
  messages?: ChatMessage[];
};

type RecommendationPlan = {
  reply: string;
  titles: string[];
};

const client = new OpenAI({
  apiKey: process.env.DEEPSEEK_API_KEY,
  baseURL: process.env.DEEPSEEK_BASE_URL || "https://api.deepseek.com",
  timeout: 30_000,
});

const model = process.env.DEEPSEEK_MODEL_NAME || "deepseek-chat";

function parseJsonObject(text: string): RecommendationPlan {
  const jsonMatch = text.match(/\{[\s\S]*\}/);

  if (!jsonMatch) {
    throw new Error("Model response did not include a JSON object");
  }

  const parsed = JSON.parse(jsonMatch[0]) as Partial<RecommendationPlan>;

  if (!Array.isArray(parsed.titles)) {
    throw new Error("Model response did not include a titles array");
  }

  return {
    reply:
      typeof parsed.reply === "string"
        ? parsed.reply
        : "I found a few movies that fit your mood.",
    titles: parsed.titles
      .filter((title): title is string => typeof title === "string")
      .map((title) => title.trim())
      .filter(Boolean)
      .slice(0, 10),
  };
}

async function createRecommendationPlan(
  messages: ChatMessage[]
): Promise<RecommendationPlan> {
  const response = await client.chat.completions.create({
    model,
    temperature: 0.4,
    response_format: { type: "json_object" },
    messages: [
      {
        role: "system",
        content:
          "You are a thoughtful movie recommendation assistant. Infer real movie titles from the user's mood, genre, language, era, and constraints. Return only JSON shaped as {\"reply\":\"brief friendly sentence\",\"titles\":[\"English movie title\", ...]}. Include 6 to 10 well-known real movie titles. Use English titles so OMDB can find them.",
      },
      ...messages.map((message) => ({
        role: message.role,
        content: message.content,
      })),
    ],
  });

  const content = response.choices[0]?.message?.content;

  if (!content) {
    throw new Error("DeepSeek returned an empty response");
  }

  return parseJsonObject(content);
}

export async function POST(request: Request) {
  try {
    const body = (await request.json()) as ChatRequest;
    const messages = body.messages?.filter(
      (message): message is ChatMessage =>
        (message.role === "user" || message.role === "assistant") &&
        typeof message.content === "string"
    );

    if (!messages?.length) {
      return Response.json(
        { error: "At least one message is required" },
        { status: 400 }
      );
    }

    const plan = await createRecommendationPlan(messages.slice(-8));
    const movies = await fetchMovies({ query: plan.titles.join(" | ") });

    return Response.json({
      reply: plan.reply,
      titles: plan.titles,
      movies,
    });
  } catch (error) {
    console.error("Chat API failed:", error);
    return Response.json(
      {
        error:
          error instanceof Error ? error.message : "Unknown chat API failure",
      },
      { status: 500 }
    );
  }
}
