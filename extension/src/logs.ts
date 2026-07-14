/** Live-tailing job logs as read-only virtual documents (gq-log: scheme). */

import * as vscode from "vscode";
import { GqApi } from "./api";

export const SCHEME = "gq-log";

interface Tail {
  jobId: number;
  text: string;
  offset: number;
  jobState: string;
}

export function logUri(jobId: number, name: string): vscode.Uri {
  // filename ends in .log so users get log syntax highlighting if installed
  const safe = name.replace(/[^\w.-]+/g, "_");
  return vscode.Uri.parse(`${SCHEME}:/${jobId}-${safe}.log`);
}

function jobIdFromUri(uri: vscode.Uri): number {
  return Number(uri.path.replace(/^\//, "").split("-")[0]);
}

export class LogTailer implements vscode.TextDocumentContentProvider {
  private _onDidChange = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this._onDidChange.event;

  private tails = new Map<string, Tail>();
  private timer: NodeJS.Timeout | undefined;

  constructor(private api: GqApi) {}

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.tails.get(uri.toString())?.text ?? "loading…";
  }

  async open(jobId: number, name: string): Promise<void> {
    const uri = logUri(jobId, name);
    if (!this.tails.has(uri.toString())) {
      this.tails.set(uri.toString(), { jobId, text: "", offset: 0, jobState: "RUNNING" });
    }
    const doc = await vscode.workspace.openTextDocument(uri);
    await vscode.window.showTextDocument(doc, { preview: true, preserveFocus: false });
    await this.poll(uri);
    this.ensureTimer();
  }

  /** Poll every open gq-log document; drop tails whose documents were closed. */
  private ensureTimer(): void {
    if (this.timer) {
      return;
    }
    this.timer = setInterval(async () => {
      const openUris = new Set(
        vscode.workspace.textDocuments
          .filter((d) => d.uri.scheme === SCHEME)
          .map((d) => d.uri.toString())
      );
      for (const key of [...this.tails.keys()]) {
        if (!openUris.has(key)) {
          this.tails.delete(key);
          continue;
        }
        await this.poll(vscode.Uri.parse(key));
      }
      if (this.tails.size === 0 && this.timer) {
        clearInterval(this.timer);
        this.timer = undefined;
      }
    }, 1000);
  }

  private async poll(uri: vscode.Uri): Promise<void> {
    const tail = this.tails.get(uri.toString());
    if (!tail) {
      return;
    }
    try {
      const chunk = await this.api.logs(tail.jobId, tail.offset);
      tail.jobState = chunk.jobState;
      if (chunk.text.length === 0) {
        return;
      }
      tail.text += chunk.text;
      tail.offset = chunk.nextOffset;
      this._onDidChange.fire(uri);
      this.autoScroll(uri);
    } catch {
      // daemon briefly unreachable: keep what we have, retry next tick
    }
  }

  private autoScroll(uri: vscode.Uri): void {
    for (const editor of vscode.window.visibleTextEditors) {
      if (editor.document.uri.toString() !== uri.toString()) {
        continue;
      }
      // only follow the bottom if the user hasn't scrolled/selected elsewhere
      const atBottom =
        editor.selection.isEmpty &&
        editor.selection.active.line >= editor.document.lineCount - 2;
      if (atBottom || editor.selection.active.line === 0) {
        const last = editor.document.lineCount - 1;
        const pos = new vscode.Position(last, 0);
        editor.revealRange(new vscode.Range(pos, pos));
        editor.selection = new vscode.Selection(pos, pos);
      }
    }
  }

  dispose(): void {
    if (this.timer) {
      clearInterval(this.timer);
    }
    this._onDidChange.dispose();
  }
}
