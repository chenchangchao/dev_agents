"use client";

import { FormEvent, useMemo, useState } from "react";
import { Loader2, Send } from "lucide-react";
import MovieCard from "@/app/_components/movie-card";
import NoMoviesCard from "@/app/_components/no-movies-card";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Movie } from "@/types";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  movies?: Movie[];
};

type ChatApiResponse = {
  reply?: string;
  movies?: Movie[];
  error?: string;
};

const starterPrompts = [
  "我刚失恋，想看一点温暖治愈的电影",
  "Recommend smart sci-fi movies with emotional depth",
  "想看轻松的爱情喜剧，最好适合周末晚上",
];

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: "assistant",
      content: "Hello! Tell me your mood, genre, language, or era, and I will recommend movies with IMDb cards.",
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const requestMessages = useMemo(
    () =>
      messages
        .filter((message) => message.role === "user" || message.role === "assistant")
        .map(({ role, content }) => ({ role, content })),
    [messages]
  );

  async function sendMessage(nextInput?: string) {
    const content = (nextInput ?? input).trim();

    if (!content || isLoading) {
      return;
    }

    const userMessage: ChatMessage = { role: "user", content };
    const nextMessages = [...messages, userMessage];

    setMessages(nextMessages);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          messages: [...requestMessages, userMessage].slice(-8),
        }),
      });
      const data = (await response.json()) as ChatApiResponse;

      if (!response.ok) {
        throw new Error(data.error || "Failed to get recommendations");
      }

      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content:
            data.reply ||
            "I found a few movies that match what you described.",
          movies: data.movies || [],
        },
      ]);
    } catch (error) {
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content:
            error instanceof Error
              ? error.message
              : "Something went wrong while getting recommendations.",
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void sendMessage();
  }

  return (
    <main className="mx-auto flex h-screen max-w-6xl flex-col px-4 py-5">
      <header className="mb-4">
        <h1 className="text-2xl font-semibold">Movie Suggestion Bot Pure</h1>
        <p className="text-sm text-muted-foreground">
          No CopilotKit. Just Next.js, DeepSeek, and OMDB.
        </p>
      </header>

      <section className="min-h-0 flex-1 overflow-y-auto rounded-lg border bg-muted/20 p-4">
        <div className="space-y-4">
          {messages.map((message, index) => (
            <div
              key={`${message.role}-${index}`}
              className={
                message.role === "user"
                  ? "flex justify-end"
                  : "flex justify-start"
              }
            >
              <div className="max-w-[90%] space-y-3">
                <Card
                  className={
                    message.role === "user"
                      ? "bg-primary text-primary-foreground"
                      : "bg-background"
                  }
                >
                  <CardContent className="px-4 py-3 text-sm leading-6">
                    {message.content}
                  </CardContent>
                </Card>

                {message.movies && (
                  message.movies.length > 0 ? (
                    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-3">
                      {message.movies.map((movie) => (
                        <MovieCard key={movie.imdbID} movie={movie} />
                      ))}
                    </div>
                  ) : (
                    <NoMoviesCard />
                  )
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <Card>
                <CardContent className="flex items-center gap-2 px-4 py-3 text-sm text-muted-foreground">
                  <Loader2 className="size-4 animate-spin" />
                  Thinking and checking OMDB...
                </CardContent>
              </Card>
            </div>
          )}
        </div>
      </section>

      <div className="mt-4 flex flex-wrap gap-2">
        {starterPrompts.map((prompt) => (
          <Button
            key={prompt}
            type="button"
            variant="outline"
            size="sm"
            onClick={() => void sendMessage(prompt)}
            disabled={isLoading}
          >
            {prompt}
          </Button>
        ))}
      </div>

      <form onSubmit={handleSubmit} className="mt-3 flex gap-2">
        <textarea
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Describe your mood or what kind of movie you want..."
          className="min-h-12 flex-1 resize-none rounded-md border bg-background px-3 py-2 text-sm outline-none focus-visible:ring-2 focus-visible:ring-ring"
          rows={2}
        />
        <Button type="submit" disabled={isLoading || !input.trim()}>
          {isLoading ? (
            <Loader2 className="size-4 animate-spin" />
          ) : (
            <Send className="size-4" />
          )}
          Send
        </Button>
      </form>
    </main>
  );
}
