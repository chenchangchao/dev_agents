export interface ScoredChunk {
  chunk: string;
  score: number;
}

import { tokenize } from "./tokenize";

function tokens(text: string): Set<string> {
  return new Set(tokenize(text));
}

export function keywordSearch(query: string, chunks: string[], topK = 3): ScoredChunk[] {
  const queryTokens = tokens(query);
  return chunks
    .map((chunk) => {
      const chunkTokens = tokens(chunk);
      let score = 0;
      for (const token of queryTokens) {
        if (chunkTokens.has(token)) score += 1;
      }
      return { chunk, score };
    })
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}
