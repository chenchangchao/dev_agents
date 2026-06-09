import { chatText } from "@agent/core";
import { bm25Search } from "./bm25";
import { chunkText } from "./chunk";
import { hybridSearch } from "./hybrid";
import { keywordSearch } from "./retriever";
import { vectorSearch } from "./vector";

export type RetrievalStrategy = "keyword" | "bm25" | "vector" | "hybrid";

export class SimpleRagAgent {
  private readonly chunks: string[];

  constructor(corpus: string) {
    this.chunks = chunkText(corpus);
  }

  retrieve(query: string, strategy: RetrievalStrategy = "keyword"): string[] {
    switch (strategy) {
      case "bm25":
        return bm25Search(query, this.chunks, 3).map((item) => item.chunk);
      case "vector":
        return vectorSearch(query, this.chunks, 3).map((item) => item.chunk);
      case "hybrid":
        return hybridSearch(query, this.chunks, 3).map((item) => item.chunk);
      default:
        return keywordSearch(query, this.chunks, 3).map((item) => item.chunk);
    }
  }

  async ask(query: string, strategy: RetrievalStrategy = "hybrid"): Promise<string> {
    const refs = this.retrieve(query, strategy);
    const prompt = `请基于参考资料回答问题。资料不足时说明需要补充信息。\n\n检索策略：${strategy}\n参考资料：\n${refs.join("\n")}\n\n问题：${query}`;
    return chatText(prompt, { temperature: 0.2, maxTokens: 360, fallbackLabel: "rag-agent" });
  }
}
