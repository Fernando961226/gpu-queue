import * as vscode from "vscode";
import { GqApi, Job } from "./api";
import { GpuTreeProvider } from "./gpus";
import { LogTailer, SCHEME } from "./logs";
import { JobItem, JobTreeProvider } from "./tree";

export function activate(context: vscode.ExtensionContext): void {
  const api = new GqApi();
  const tree = new JobTreeProvider();
  const gpuTree = new GpuTreeProvider();
  const tailer = new LogTailer(api);
  const notifier = new CompletionNotifier(api);

  const statusBar = vscode.window.createStatusBarItem(vscode.StatusBarAlignment.Left, 50);
  statusBar.command = "gpuQueueJobs.focus";
  statusBar.show();

  context.subscriptions.push(
    vscode.window.registerTreeDataProvider("gpuQueueJobs", tree),
    vscode.window.registerTreeDataProvider("gpuQueueGpus", gpuTree),
    vscode.workspace.registerTextDocumentContentProvider(SCHEME, tailer),
    statusBar,
    tailer,

    vscode.commands.registerCommand("gpuQueue.refresh", () => refresh()),
    vscode.commands.registerCommand(
      "gpuQueue.showLogs",
      async (arg: number | JobItem, name?: string) => {
        const jobId = typeof arg === "number" ? arg : arg.job.id;
        const jobName = typeof arg === "number" ? name ?? String(arg) : arg.job.name;
        await tailer.open(jobId, jobName);
      }
    ),
    vscode.commands.registerCommand("gpuQueue.cancel", async (item: JobItem) => {
      try {
        await api.cancel(item.job.id);
        void refresh();
      } catch (e) {
        void vscode.window.showErrorMessage(`gq cancel failed: ${message(e)}`);
      }
    }),
    vscode.commands.registerCommand("gpuQueue.requeue", async (item: JobItem) => {
      try {
        await api.requeue(item.job.id);
        void refresh();
      } catch (e) {
        void vscode.window.showErrorMessage(`gq requeue failed: ${message(e)}`);
      }
    })
  );

  let timer: NodeJS.Timeout | undefined;

  async function refresh(): Promise<void> {
    const config = vscode.workspace.getConfiguration("gpuQueue");
    const finishedLimit = config.get<number>("finishedLimit", 15);
    try {
      const [active, finished, gpus] = await Promise.all([
        api.listJobs(["QUEUED", "RUNNING"]),
        api.listJobs(["DONE", "FAILED", "CANCELLED"], finishedLimit),
        api.gpus(),
      ]);
      const jobs = dedupe([...active, ...finished]);
      tree.update(jobs, false);
      gpuTree.update(gpus, jobs, false);
      notifier.observe(jobs);

      const running = jobs.filter((j) => j.state === "RUNNING").length;
      const queued = jobs.filter((j) => j.state === "QUEUED").length;
      const free = gpus.filter((g) => g.state === "free").length;
      statusBar.text = `$(server-process) gq: ${running}▶ ${queued}⏳ · ${free}/${gpus.length} free`;
      statusBar.tooltip = `gpu-queue: ${running} running, ${queued} queued, ${free} of ${gpus.length} GPUs free`;
    } catch {
      tree.update([], true);
      gpuTree.update([], [], true);
      statusBar.text = "$(debug-disconnect) gq: offline";
      statusBar.tooltip = "gpu-queue daemon not reachable — run `gq daemon start`";
    }
  }

  function schedule(): void {
    if (timer) {
      clearInterval(timer);
    }
    const ms = vscode.workspace
      .getConfiguration("gpuQueue")
      .get<number>("pollIntervalMs", 2000);
    timer = setInterval(() => void refresh(), Math.max(500, ms));
  }

  context.subscriptions.push(
    vscode.workspace.onDidChangeConfiguration((e) => {
      if (e.affectsConfiguration("gpuQueue")) {
        schedule();
        void refresh();
      }
    }),
    { dispose: () => timer && clearInterval(timer) }
  );

  schedule();
  void refresh();
}

function dedupe(jobs: Job[]): Job[] {
  const seen = new Map<number, Job>();
  for (const j of jobs) {
    seen.set(j.id, j);
  }
  return [...seen.values()];
}

function message(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Fires a notification when a job we saw RUNNING/QUEUED reaches a final state. */
class CompletionNotifier {
  private lastState = new Map<number, string>();

  constructor(private api: GqApi) {}

  observe(jobs: Job[]): void {
    const notify = vscode.workspace
      .getConfiguration("gpuQueue")
      .get<boolean>("notifyOnCompletion", true);
    for (const job of jobs) {
      const prev = this.lastState.get(job.id);
      this.lastState.set(job.id, job.state);
      if (!notify || prev === undefined || prev === job.state) {
        continue;
      }
      if (prev === "RUNNING" && (job.state === "DONE" || job.state === "FAILED")) {
        void this.announce(job);
      }
    }
    // forget jobs that scrolled out of the visible window
    const visible = new Set(jobs.map((j) => j.id));
    for (const id of [...this.lastState.keys()]) {
      if (!visible.has(id)) {
        this.lastState.delete(id);
      }
    }
  }

  private async announce(job: Job): Promise<void> {
    const label = `#${job.id} ${job.name}`;
    const action =
      job.state === "DONE"
        ? await vscode.window.showInformationMessage(`gq: ${label} finished`, "Open logs")
        : await vscode.window.showErrorMessage(
            `gq: ${label} failed (exit ${job.exit_code ?? "?"})`,
            "Open logs",
            "Requeue"
          );
    if (action === "Open logs") {
      void vscode.commands.executeCommand("gpuQueue.showLogs", job.id, job.name);
    } else if (action === "Requeue") {
      try {
        await this.api.requeue(job.id);
      } catch (e) {
        void vscode.window.showErrorMessage(`gq requeue failed: ${message(e)}`);
      }
    }
  }
}

export function deactivate(): void {}
