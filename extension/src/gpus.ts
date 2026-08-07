/** The GPUs tree: one row per GPU with a utilisation bar and its owning job. */

import * as vscode from "vscode";
import { Gpu, Job, Tenant } from "./api";

const BAR_CELLS = 10;
const SPARK = "▁▂▃▄▅▆▇█";

function clampPct(pct: number): number {
  return Math.max(0, Math.min(100, Math.round(pct)));
}

function bar(pct: number): string {
  const filled = Math.round((clampPct(pct) / 100) * BAR_CELLS);
  return "█".repeat(filled) + "░".repeat(BAR_CELLS - filled);
}

function sparkline(values: number[]): string {
  if (!values.length) return "";
  return values
    .map((v) => SPARK[Math.min(SPARK.length - 1, Math.floor((clampPct(v) / 100) * SPARK.length))])
    .join("");
}

function gib(mb: number): string {
  return (mb / 1024).toFixed(1);
}

function tenantLabel(t: Tenant): string {
  const budget = t.vram_mb === null ? "" : ` [${gib(t.vram_mb)}G]`;
  return `#${t.job_id} ${t.name}${budget}`;
}

function owner(gpu: Gpu): string {
  if (gpu.state === "external") return "external";
  const tenants = gpu.tenants ?? [];
  if (!tenants.length) return "free";
  // One tenant reads better named; several would overflow the row.
  return tenants.length === 1 ? tenantLabel(tenants[0]) : `${tenants.length} jobs`;
}

/** Fraction of schedulable VRAM already reserved, 0-100. */
function reservedPct(gpu: Gpu): number {
  if (!gpu.vram_capacity_mb) return 0;
  return (gpu.vram_reserved_mb / gpu.vram_capacity_mb) * 100;
}

function icon(gpu: Gpu): vscode.ThemeIcon {
  switch (gpu.state) {
    case "allocated":
      return new vscode.ThemeIcon("circle-filled", new vscode.ThemeColor("charts.blue"));
    case "external":
      return new vscode.ThemeIcon("warning", new vscode.ThemeColor("charts.orange"));
    case "free":
      return new vscode.ThemeIcon("circle-outline");
  }
}

export class GpuItem extends vscode.TreeItem {
  constructor(public readonly gpu: Gpu, history: number[], jobs: Map<number, Job>) {
    super(`GPU ${gpu.index}`, vscode.TreeItemCollapsibleState.None);
    this.id = `gpu-${gpu.index}`;
    this.description =
      `${bar(gpu.util_pct)} ${clampPct(gpu.util_pct)}%` +
      ` · ${gib(gpu.mem_used_mb)}/${gib(gpu.mem_total_mb)} GiB` +
      ` · ${owner(gpu)}`;
    this.tooltip = tooltip(gpu, history, jobs);
    this.iconPath = icon(gpu);
    this.contextValue = `gpu-${gpu.state}`;
  }
}

function tooltip(gpu: Gpu, history: number[], jobs: Map<number, Job>): vscode.MarkdownString {
  const md = new vscode.MarkdownString();
  md.appendMarkdown(`**GPU ${gpu.index}** · ${gpu.name}\n\n`);
  const spark = sparkline(history);
  md.appendMarkdown(`util: \`${spark}\` ${clampPct(gpu.util_pct)}%\n\n`);
  md.appendMarkdown(`vram: ${gib(gpu.mem_used_mb)} / ${gib(gpu.mem_total_mb)} GiB\n\n`);
  if (gpu.state === "external") {
    md.appendMarkdown(`in use by another user — gq will not schedule here\n`);
    return md;
  }
  const tenants = gpu.tenants ?? [];
  if (gpu.vram_capacity_mb) {
    // Reserved is what jobs *declared*, which is what the scheduler allocates
    // against — deliberately not the live NVML reading above.
    md.appendMarkdown(
      `reserved: \`${bar(reservedPct(gpu))}\` ` +
        `${gib(gpu.vram_reserved_mb)} / ${gib(gpu.vram_capacity_mb)} GiB\n\n`
    );
  }
  if (!tenants.length) {
    md.appendMarkdown(`free\n`);
  } else if (tenants.some((t) => t.vram_mb === null)) {
    md.appendMarkdown(`exclusive: ${tenants.map(tenantLabel).join(", ")}\n`);
  } else {
    md.appendMarkdown(`sharing (${tenants.length}):\n\n`);
    for (const t of tenants) {
      md.appendMarkdown(`- ${tenantLabel(t)}\n`);
    }
  }
  return md;
}

export class GpuTreeProvider implements vscode.TreeDataProvider<vscode.TreeItem> {
  private _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  private gpus: Gpu[] = [];
  private jobs = new Map<number, Job>();
  private offline = false;
  /** gpu index -> recent util_pct samples, oldest first */
  private history = new Map<number, number[]>();

  update(gpus: Gpu[], jobs: Job[], offline: boolean): void {
    this.gpus = gpus;
    this.jobs = new Map(jobs.map((j) => [j.id, j]));
    this.offline = offline;
    if (!offline) {
      this.record(gpus);
    }
    this._onDidChange.fire();
  }

  private record(gpus: Gpu[]): void {
    const cap = Math.max(
      2,
      vscode.workspace.getConfiguration("gpuQueue").get<number>("gpuHistoryLength", 30)
    );
    for (const g of gpus) {
      const samples = this.history.get(g.index) ?? [];
      samples.push(g.util_pct);
      // trim from the front so the sparkline reads oldest -> newest
      this.history.set(g.index, samples.slice(Math.max(0, samples.length - cap)));
    }
    // drop GPUs that vanished (fake-gpu dev mode, driver reload)
    const present = new Set(gpus.map((g) => g.index));
    for (const idx of [...this.history.keys()]) {
      if (!present.has(idx)) {
        this.history.delete(idx);
      }
    }
  }

  getTreeItem(element: vscode.TreeItem): vscode.TreeItem {
    return element;
  }

  getChildren(element?: vscode.TreeItem): vscode.TreeItem[] {
    if (element) {
      return [];
    }
    if (this.offline) {
      const item = new vscode.TreeItem("daemon offline — run `gq daemon start`");
      item.iconPath = new vscode.ThemeIcon("debug-disconnect");
      return [item];
    }
    return this.gpus.map((g) => new GpuItem(g, this.history.get(g.index) ?? [], this.jobs));
  }
}
