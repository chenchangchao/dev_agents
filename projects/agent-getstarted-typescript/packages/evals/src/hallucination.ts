export function roughSimilarity(a: string, b: string): number {
  const left = new Set(a.toLowerCase().match(/[a-z0-9]+|[\u4e00-\u9fa5]/g) ?? []);
  const right = new Set(b.toLowerCase().match(/[a-z0-9]+|[\u4e00-\u9fa5]/g) ?? []);
  if (!left.size || !right.size) return 0;
  let overlap = 0;
  for (const token of left) {
    if (right.has(token)) overlap += 1;
  }
  return overlap / Math.max(left.size, right.size);
}

export function isLikelyHallucination(answer: string, reference: string, threshold = 0.25): boolean {
  return roughSimilarity(answer, reference) < threshold;
}
