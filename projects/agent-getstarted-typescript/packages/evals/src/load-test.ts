export interface LoadTestResult<T> {
  total: number;
  success: number;
  failed: number;
  avgMs: number;
  qps: number;
  samples: T[];
}

export async function runLoadTest<T>(
  total: number,
  concurrency: number,
  worker: (index: number) => Promise<T>
): Promise<LoadTestResult<T>> {
  const started = Date.now();
  const results: T[] = [];
  let cursor = 0;
  let failed = 0;
  const durations: number[] = [];

  async function runner(): Promise<void> {
    while (cursor < total) {
      const index = cursor;
      cursor += 1;
      const itemStarted = Date.now();
      try {
        results.push(await worker(index));
      } catch {
        failed += 1;
      } finally {
        durations.push(Date.now() - itemStarted);
      }
    }
  }

  await Promise.all(Array.from({ length: Math.min(total, concurrency) }, () => runner()));
  const elapsed = Math.max(1, Date.now() - started);
  const success = total - failed;
  return {
    total,
    success,
    failed,
    avgMs: durations.reduce((sum, value) => sum + value, 0) / durations.length,
    qps: total / (elapsed / 1000),
    samples: results.slice(0, 5)
  };
}
