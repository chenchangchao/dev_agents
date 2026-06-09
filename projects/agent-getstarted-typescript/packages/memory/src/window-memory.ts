import type { Message } from "@agent/core";

export class WindowMemory {
  private messages: Message[] = [];

  constructor(private readonly maxMessages = 6) {}

  add(role: Message["role"], content: string): void {
    this.messages.push({ role, content });
    if (this.messages.length > this.maxMessages) {
      this.messages = this.messages.slice(-this.maxMessages);
    }
  }

  all(): Message[] {
    return [...this.messages];
  }

  text(): string {
    return this.messages.map((message) => `${message.role}: ${message.content}`).join("\n");
  }
}
