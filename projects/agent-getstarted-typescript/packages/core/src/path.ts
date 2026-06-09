import { mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";

export const PROJECT_ROOT = resolve(import.meta.dir, "../../..");
export const DATA_ROOT = resolve(PROJECT_ROOT, "data");

export function projectPath(...parts: string[]): string {
  return resolve(PROJECT_ROOT, ...parts);
}

export function dataPath(...parts: string[]): string {
  const fullPath = resolve(DATA_ROOT, ...parts);
  mkdirSync(dirname(fullPath), { recursive: true });
  return fullPath;
}

export async function ensureDir(path: string): Promise<void> {
  mkdirSync(path, { recursive: true });
}
