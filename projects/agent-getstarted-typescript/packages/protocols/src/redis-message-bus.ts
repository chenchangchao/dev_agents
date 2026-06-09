import Redis from "ioredis";
import type { A2AMessage } from "./a2a";
import type { MessageBus, MessageHandler } from "./message-bus";

export interface RedisMessageBusOptions {
  url?: string;
  channelPrefix?: string;
}

export class RedisMessageBus implements MessageBus {
  private readonly pub: Redis;
  private readonly sub: Redis;
  private readonly channelPrefix: string;
  private readonly handlers = new Map<string, Set<MessageHandler>>();
  private started = false;

  constructor(options: RedisMessageBusOptions = {}) {
    const url = options.url ?? Bun.env.REDIS_URL ?? "redis://127.0.0.1:6379";
    this.channelPrefix = options.channelPrefix ?? Bun.env.REDIS_CHANNEL_PREFIX ?? "agent-ts";
    const common = { lazyConnect: true, maxRetriesPerRequest: 1, retryStrategy: () => null };
    this.pub = new Redis(url, common);
    this.sub = new Redis(url, common);
    this.pub.on("error", () => undefined);
    this.sub.on("error", () => undefined);
  }

  private topicName(topic: string): string {
    return `${this.channelPrefix}:${topic}`;
  }

  private async ensureStarted(): Promise<void> {
    if (this.started) return;
    await this.pub.connect().catch(() => undefined);
    await this.sub.connect().catch(() => undefined);
    this.sub.on("message", (channel, payload) => {
      const topic = channel.replace(`${this.channelPrefix}:`, "");
      const handlers = this.handlers.get(topic);
      if (!handlers?.size) return;
      const parsed = JSON.parse(payload) as A2AMessage<unknown>;
      for (const handler of handlers) {
        void handler(parsed);
      }
    });
    this.started = true;
  }

  async publish<T>(topic: string, message: A2AMessage<T>): Promise<void> {
    await this.ensureStarted();
    await this.pub.publish(this.topicName(topic), JSON.stringify(message));
  }

  subscribe<T>(topic: string, handler: MessageHandler<T>): () => void {
    const handlers = this.handlers.get(topic) ?? new Set<MessageHandler>();
    handlers.add(handler as MessageHandler);
    this.handlers.set(topic, handlers);
    void this.ensureStarted().then(() => this.sub.subscribe(this.topicName(topic)));
    return () => {
      handlers.delete(handler as MessageHandler);
      if (handlers.size === 0) {
        this.handlers.delete(topic);
        void this.sub.unsubscribe(this.topicName(topic));
      }
    };
  }

  async close(): Promise<void> {
    await Promise.allSettled([this.pub.quit(), this.sub.quit()]);
  }
}
