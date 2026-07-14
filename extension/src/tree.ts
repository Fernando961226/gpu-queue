/** The Jobs tree: Running / Queued / Finished sections with per-job items. */

import * as vscode from "vscode";
import { Job } from "./api";

type Element = SectionItem | JobItem;

const SECTIONS = ["Running", "Queued", "Finished"] as const;
type Section = (typeof SECTIONS)[number];

export class SectionItem extends vscode.TreeItem {
  constructor(public readonly section: Section, count: number) {
    super(`${section} (${count})`, vscode.TreeItemCollapsibleState.Expanded);
    this.contextValue = "section";
    this.id = `section-${section}`;
  }
}

export class JobItem extends vscode.TreeItem {
  constructor(public readonly job: Job) {
    super(`#${job.id} ${job.name}`, vscode.TreeItemCollapsibleState.None);
    this.id = `job-${job.id}`;
    this.description = describe(job);
    this.tooltip = tooltip(job);
    this.iconPath = icon(job);
    this.contextValue =
      job.state === "RUNNING" ? "job-running"
      : job.state === "QUEUED" ? "job-queued"
      : "job-finished";
    this.command = {
      command: "gpuQueue.showLogs",
      title: "Show Logs",
      arguments: [job.id],
    };
  }
}

function fmtDuration(seconds: number): string {
  const s = Math.max(0, Math.floor(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m${String(s % 60).padStart(2, "0")}s`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h${String(m % 60).padStart(2, "0")}m`;
  return `${Math.floor(h / 24)}d${String(h % 24).padStart(2, "0")}h`;
}

function runtime(job: Job): string {
  if (job.started_at === null) return "";
  const end = job.finished_at ?? Date.now() / 1000;
  return fmtDuration(end - job.started_at);
}

function describe(job: Job): string {
  const parts: string[] = [];
  if (job.state === "RUNNING" && job.gpu_ids) {
    parts.push(`gpu ${job.gpu_ids.join(",")}`);
    parts.push(runtime(job));
  } else if (job.state === "QUEUED") {
    parts.push(`wants ${job.gpus_requested} gpu${job.gpus_requested === 1 ? "" : "s"}`);
  } else {
    parts.push(job.state.toLowerCase());
    if (job.exit_code !== null && job.exit_code !== 0) {
      parts.push(`exit ${job.exit_code}`);
    }
    const rt = runtime(job);
    if (rt) parts.push(rt);
  }
  return parts.filter(Boolean).join(" · ");
}

function tooltip(job: Job): vscode.MarkdownString {
  const md = new vscode.MarkdownString();
  md.appendMarkdown(`**#${job.id} ${job.name}** — ${job.state}\n\n`);
  md.appendCodeblock(job.command.join(" "), "bash");
  md.appendMarkdown(`\ncwd: \`${job.workdir}\`\n\n`);
  if (job.conda_env) md.appendMarkdown(`conda: \`${job.conda_env}\`\n\n`);
  if (job.note) md.appendMarkdown(`note: ${job.note}\n`);
  return md;
}

function icon(job: Job): vscode.ThemeIcon {
  switch (job.state) {
    case "RUNNING":
      return new vscode.ThemeIcon("play", new vscode.ThemeColor("charts.green"));
    case "QUEUED":
      return new vscode.ThemeIcon("watch", new vscode.ThemeColor("charts.yellow"));
    case "DONE":
      return new vscode.ThemeIcon("check", new vscode.ThemeColor("charts.green"));
    case "FAILED":
      return new vscode.ThemeIcon("error", new vscode.ThemeColor("charts.red"));
    case "CANCELLED":
      return new vscode.ThemeIcon("circle-slash");
  }
}

export class JobTreeProvider implements vscode.TreeDataProvider<Element> {
  private _onDidChange = new vscode.EventEmitter<void>();
  readonly onDidChangeTreeData = this._onDidChange.event;

  private jobs: Job[] = [];
  private offline = false;

  update(jobs: Job[], offline: boolean): void {
    this.jobs = jobs;
    this.offline = offline;
    this._onDidChange.fire();
  }

  getTreeItem(element: Element): vscode.TreeItem {
    return element;
  }

  getChildren(element?: Element): Element[] {
    if (element instanceof SectionItem) {
      return this.inSection(element.section).map((j) => new JobItem(j));
    }
    if (element) {
      return [];
    }
    if (this.offline) {
      const item = new vscode.TreeItem("daemon offline — run `gq daemon start`");
      item.iconPath = new vscode.ThemeIcon("debug-disconnect");
      return [item as Element];
    }
    return SECTIONS.map((s) => new SectionItem(s, this.inSection(s).length));
  }

  private inSection(section: Section): Job[] {
    switch (section) {
      case "Running":
        return this.jobs.filter((j) => j.state === "RUNNING");
      case "Queued":
        return this.jobs.filter((j) => j.state === "QUEUED");
      case "Finished":
        return this.jobs.filter(
          (j) => j.state === "DONE" || j.state === "FAILED" || j.state === "CANCELLED"
        );
    }
  }
}
