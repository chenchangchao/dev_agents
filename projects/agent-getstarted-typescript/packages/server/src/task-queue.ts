export type TaskStatus = "queued" | "running" | "done" | "failed";

export interface TaskRecord<T = unknown> {
  id: string;
  status: TaskStatus;
  input: T;
  result?: unknown;
  error?: string;
  createdAt: number;
  updatedAt: number;
}

export class AsyncTaskQueue<T = unknown> {
  private readonly tasks = new Map<string, TaskRecord<T>>();
  private running = false;

  constructor(private readonly worker: (input: T) => Promise<unknown>) {}

  enqueue(input: T): TaskRecord<T> {
    const now = Date.now();
    const task: TaskRecord<T> = {
      id: crypto.randomUUID(),
      status: "queued",
      input,
      createdAt: now,
      updatedAt: now
    };
    this.tasks.set(task.id, task);
    void this.drain();
    return task;
  }

  get(id: string): TaskRecord<T> | undefined {
    return this.tasks.get(id);
  }

  list(): TaskRecord<T>[] {
    return [...this.tasks.values()];
  }

  private async drain(): Promise<void> {
    if (this.running) return;
    this.running = true;
    try {
      for (const task of this.tasks.values()) {
        if (task.status !== "queued") continue;
        task.status = "running";
        task.updatedAt = Date.now();
        try {
          task.result = await this.worker(task.input);
          task.status = "done";
        } catch (error) {
          task.error = error instanceof Error ? error.message : String(error);
          task.status = "failed";
        } finally {
          task.updatedAt = Date.now();
        }
      }
    } finally {
      this.running = false;
    }
  }
}
