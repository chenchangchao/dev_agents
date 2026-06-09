export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetAt: number;
}

interface Bucket {
  count: number;
  resetAt: number;
}

export class FixedWindowRateLimiter {
  private readonly buckets = new Map<string, Bucket>();

  constructor(
    private readonly limit = 20,
    private readonly windowMs = 60_000
  ) {}

  check(key: string, now = Date.now()): RateLimitResult {
    const current = this.buckets.get(key);
    if (!current || current.resetAt <= now) {
      const bucket = { count: 1, resetAt: now + this.windowMs };
      this.buckets.set(key, bucket);
      return { allowed: true, remaining: this.limit - 1, resetAt: bucket.resetAt };
    }

    if (current.count >= this.limit) {
      return { allowed: false, remaining: 0, resetAt: current.resetAt };
    }

    current.count += 1;
    return { allowed: true, remaining: this.limit - current.count, resetAt: current.resetAt };
  }
}
