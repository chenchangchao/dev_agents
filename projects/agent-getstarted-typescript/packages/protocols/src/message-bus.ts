import type { A2AMessage } from "./a2a";

export type MessageHandler<T = unknown> = (message: A2AMessage<T>) => void | Promise<void>;

export interface MessageBus {
  publish<T>(topic: string, message: A2AMessage<T>): Promise<void>;
  subscribe<T>(topic: string, handler: MessageHandler<T>): () => void;
}

export class InMemoryMessageBus implements MessageBus {
  private readonly handlers = new Map<string, Set<MessageHandler>>();

  async publish<T>(topic: string, message: A2AMessage<T>): Promise<void> {
    const handlers = this.handlers.get(topic) ?? new Set();
    await Promise.all([...handlers].map((handler) => handler(message)));
  }

  subscribe<T>(topic: string, handler: MessageHandler<T>): () => void {
    const handlers = this.handlers.get(topic) ?? new Set();
    handlers.add(handler as MessageHandler);
    this.handlers.set(topic, handlers);
    return () => handlers.delete(handler as MessageHandler);
  }
}
