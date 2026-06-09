import { tokenize } from "./tokenize";

export interface DenseVectorResult {
  chunk: string;
  score: number;
}

const HASH_SIZE = 64;

export function embedText(text: string): number[] {
  const vector = Array.from({ length: HASH_SIZE }, () => 0);
  for (const token of tokenize(text)) {
    let hash = 0;
    for (let index = 0; index < token.length; index += 1) {
      hash = (hash * 31 + token.charCodeAt(index)) % HASH_SIZE;
    }
    vector[hash] += 1;
  }
  return normalize(vector);
}

export function cosineSimilarity(left: number[], right: number[]): number {
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < Math.max(left.length, right.length); index += 1) {
    const a = left[index] ?? 0;
    const b = right[index] ?? 0;
    dot += a * b;
    leftNorm += a * a;
    rightNorm += b * b;
  }
  if (!leftNorm || !rightNorm) return 0;
  return dot / (Math.sqrt(leftNorm) * Math.sqrt(rightNorm));
}

export function vectorSearch(query: string, chunks: string[], topK = 3): DenseVectorResult[] {
  const queryEmbedding = embedText(query);
  return chunks
    .map((chunk) => ({ chunk, score: cosineSimilarity(queryEmbedding, embedText(chunk)) }))
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}

function normalize(values: number[]): number[] {
  const norm = Math.sqrt(values.reduce((sum, value) => sum + value * value, 0));
  if (!norm) return values;
  return values.map((value) => value / norm);
}
