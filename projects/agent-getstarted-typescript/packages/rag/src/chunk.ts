export function chunkText(text: string, chunkSize = 180, overlap = 30): string[] {
  const normalized = text.replace(/\s+/g, " ").trim();
  if (!normalized) return [];
  const chunks: string[] = [];
  let start = 0;
  while (start < normalized.length) {
    chunks.push(normalized.slice(start, start + chunkSize));
    start += Math.max(1, chunkSize - overlap);
  }
  return chunks;
}
