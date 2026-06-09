import { bm25Search } from "./bm25";
import { vectorSearch } from "./vector";

export interface HybridResult {
  chunk: string;
  score: number;
  bm25: number;
  vector: number;
}

export function hybridSearch(query: string, chunks: string[], topK = 3): HybridResult[] {
  const bm25 = bm25Search(query, chunks, chunks.length);
  const vector = vectorSearch(query, chunks, chunks.length);
  const index = new Map<string, HybridResult>();

  for (const item of bm25) {
    index.set(item.chunk, { chunk: item.chunk, score: 0, bm25: item.score, vector: 0 });
  }
  for (const item of vector) {
    const current = index.get(item.chunk) ?? { chunk: item.chunk, score: 0, bm25: 0, vector: 0 };
    current.vector = item.score;
    index.set(item.chunk, current);
  }

  const bm25Max = Math.max(...bm25.map((item) => item.score), 1);
  const vectorMax = Math.max(...vector.map((item) => item.score), 1);
  return [...index.values()]
    .map((item) => ({
      ...item,
      score: item.bm25 / bm25Max * 0.6 + item.vector / vectorMax * 0.4
    }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}
