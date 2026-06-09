export interface A2AMessage<T = unknown> {
  id: string;
  from: string;
  to: string;
  type: "request" | "response" | "event";
  payload: T;
  createdAt: number;
}

export function createA2AMessage<T>(input: Omit<A2AMessage<T>, "id" | "createdAt">): A2AMessage<T> {
  return {
    ...input,
    id: crypto.randomUUID(),
    createdAt: Date.now()
  };
}
