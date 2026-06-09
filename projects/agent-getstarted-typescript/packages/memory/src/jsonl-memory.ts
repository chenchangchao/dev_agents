import { appendFile, mkdir } from "node:fs/promises";
import { dirname } from "node:path";
import type { Message } from "@agent/core";

export class JsonlMemory {
  constructor(private readonly filePath: string) {}

  async append(message: Message): Promise<void> {
    await mkdir(dirname(this.filePath), { recursive: true });
    await appendFile(this.filePath, `${JSON.stringify({ ...message, ts: Date.now() })}\n`, "utf8");
  }

  async readAll(): Promise<Message[]> {
    const file = Bun.file(this.filePath);
    if (!(await file.exists())) return [];
    const lines = (await file.text()).split("\n").filter(Boolean);
    return lines.map((line) => JSON.parse(line) as Message);
  }
}
