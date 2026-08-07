/** Client for the gpu-queue daemon's localhost HTTP API. */

import * as vscode from "vscode";

export interface Job {
  id: number;
  name: string;
  command: string[];
  workdir: string;
  conda_env: string | null;
  gpus_requested: number;
  gpu_ids: number[] | null;
  pid: number | null;
  state: "QUEUED" | "RUNNING" | "DONE" | "FAILED" | "CANCELLED";
  submitted_at: number;
  started_at: number | null;
  finished_at: number | null;
  exit_code: number | null;
  log_path: string | null;
  note: string | null;
  /** Declared VRAM budget; null means the job holds whole GPUs exclusively. */
  vram_mb: number | null;
}

export interface Tenant {
  job_id: number;
  name: string;
  vram_mb: number | null;
}

export interface Gpu {
  index: number;
  name: string;
  state: "free" | "allocated" | "external";
  /** First tenant, kept for compatibility; prefer `tenants`. */
  job_id: number | null;
  tenants: Tenant[];
  mem_used_mb: number;
  mem_total_mb: number;
  util_pct: number;
  /** Schedulable VRAM (total minus headroom) and how much is reserved. */
  vram_capacity_mb: number;
  vram_reserved_mb: number;
}

export interface LogChunk {
  /** Raw bytes: chunk boundaries can split UTF-8 sequences, so decoding is
   * the caller's job (with a streaming TextDecoder). */
  bytes: Uint8Array;
  nextOffset: number;
  jobState: string;
}

function baseUrl(): string {
  const port = vscode.workspace.getConfiguration("gpuQueue").get<number>("port", 47563);
  return `http://127.0.0.1:${port}`;
}

async function getJson<T>(path: string): Promise<T> {
  const resp = await fetch(baseUrl() + path);
  if (!resp.ok) {
    throw new Error(await errorMessage(resp));
  }
  return (await resp.json()) as T;
}

async function errorMessage(resp: Response): Promise<string> {
  try {
    const body = (await resp.json()) as { error?: string };
    return body.error ?? `HTTP ${resp.status}`;
  } catch {
    return `HTTP ${resp.status}`;
  }
}

export class GqApi {
  async health(): Promise<boolean> {
    try {
      await getJson("/api/health");
      return true;
    } catch {
      return false;
    }
  }

  listJobs(states?: string[], limit?: number): Promise<Job[]> {
    const params = new URLSearchParams();
    if (states && states.length) {
      params.set("state", states.join(","));
    }
    if (limit) {
      params.set("limit", String(limit));
    }
    const qs = params.toString();
    return getJson<Job[]>(`/api/jobs${qs ? "?" + qs : ""}`);
  }

  gpus(): Promise<Gpu[]> {
    return getJson<Gpu[]>("/api/gpus");
  }

  async cancel(id: number): Promise<Job> {
    return this.post<Job>(`/api/jobs/${id}/cancel`);
  }

  async requeue(id: number): Promise<Job> {
    return this.post<Job>(`/api/jobs/${id}/requeue`);
  }

  async logs(id: number, offset: number): Promise<LogChunk> {
    const resp = await fetch(baseUrl() + `/api/jobs/${id}/logs?offset=${offset}`);
    if (!resp.ok) {
      throw new Error(await errorMessage(resp));
    }
    const bytes = new Uint8Array(await resp.arrayBuffer());
    return {
      bytes,
      nextOffset: Number(resp.headers.get("X-Log-Offset") ?? offset),
      jobState: resp.headers.get("X-Job-State") ?? "",
    };
  }

  private async post<T>(path: string): Promise<T> {
    const resp = await fetch(baseUrl() + path, { method: "POST" });
    if (!resp.ok) {
      throw new Error(await errorMessage(resp));
    }
    return (await resp.json()) as T;
  }
}
