import { Database } from "bun:sqlite";
import type { Message } from "@agent/core";

export class SqliteMemory {
  private readonly db: Database;

  constructor(private readonly filePath: string) {
    this.db = new Database(filePath, { create: true });
    this.db.run(`
      create table if not exists messages (
        id integer primary key autoincrement,
        session_id text not null,
        role text not null,
        content text not null,
        ts integer not null
      )
    `);
  }

  append(sessionId: string, message: Message): void {
    this.db
      .query("insert into messages (session_id, role, content, ts) values (?, ?, ?, ?)")
      .run(sessionId, message.role, message.content, Date.now());
  }

  read(sessionId: string, limit = 20): Message[] {
    return this.db
      .query("select role, content from messages where session_id = ? order by id desc limit ?")
      .all(sessionId, limit)
      .reverse() as Message[];
  }

  close(): void {
    this.db.close();
  }
}
