import { tokenize } from "./tokenize";

export interface RankedChunk {
  chunk: string;
  score: number;
}

export function bm25Search(query: string, chunks: string[], topK = 3, k1 = 1.5, b = 0.75): RankedChunk[] {
  const documents = chunks.map((chunk) => tokenize(chunk));
  const queryTerms = tokenize(query);
  const avgDocLength = documents.reduce((sum, doc) => sum + doc.length, 0) / Math.max(documents.length, 1);

  const docFreq = new Map<string, number>();
  for (const doc of documents) {
    for (const term of new Set(doc)) {
      docFreq.set(term, (docFreq.get(term) ?? 0) + 1);
    }
  }

  return chunks
    .map((chunk, index) => {
      const doc = documents[index];
      const termFreq = new Map<string, number>();
      for (const term of doc) termFreq.set(term, (termFreq.get(term) ?? 0) + 1);

      let score = 0;
      for (const term of queryTerms) {
        const freq = termFreq.get(term) ?? 0;
        if (freq === 0) continue;
        const df = docFreq.get(term) ?? 0;
        const idf = Math.log(1 + (documents.length - df + 0.5) / (df + 0.5));
        const numerator = freq * (k1 + 1);
        const denominator = freq + k1 * (1 - b + b * (doc.length / Math.max(avgDocLength, 1)));
        score += idf * (numerator / denominator);
      }
      return { chunk, score };
    })
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
}
