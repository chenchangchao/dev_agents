import Redis from "ioredis";
import type { Message } from "@agent/core";

export class RedisMemory {
  private readonly redis: Redis;
  private readonly prefix: string;

  constructor(url = Bun.env.REDIS_URL ?? "redis://127.0.0.1:6379", prefix = "agent-memory") {
    this.redis = new Redis(url, { lazyConnect: true, maxRetriesPerRequest: 1, retryStrategy: () => null });
    this.prefix = prefix;
    this.redis.on("error", () => undefined);
  }

  private key(sessionId: string): string {
    return `${this.prefix}:${sessionId}`;
  }

  private async ensureConnected(): Promise<void> {
    if (this.redis.status === "wait") {
      await this.redis.connect();
    }
  }

  async append(sessionId: string, message: Message): Promise<void> {
    await this.ensureConnected();
    await this.redis.rpush(this.key(sessionId), JSON.stringify({ ...message, ts: Date.now() }));
  }

  async read(sessionId: string, limit = 20): Promise<Message[]> {
    await this.ensureConnected();
    const items = await this.redis.lrange(this.key(sessionId), -limit, -1);
    return items.map((item) => JSON.parse(item) as Message);
  }

  async close(): Promise<void> {
    await this.redis.quit();
  }
}
