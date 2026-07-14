/** Live-tailing job logs as read-only virtual documents (gq-log: scheme). */

import { TextDecoder } from "util";
import * as vscode from "vscode";
import { GqApi } from "./api";

export const SCHEME = "gq-log";

// Finished jobs are still polled, but slowly — a requeue appends to the same
// log file, so a stopped tail has to notice the job coming back to life.
const IDLE_POLL_MS = 15000;
const ERROR_RETRY_MS = 2000;
// Catch-up cap per poll: 40 chunks × 512KB = 20MB before yielding to the timer.
const MAX_CHUNKS_PER_POLL = 40;

interface Tail {
  jobId: number;
  text: string;
  offset: number;
  jobState: string;
  polling: boolean;
  nextPollAt: number;
  decoder: TextDecoder; // streaming: chunk boundaries can split UTF-8 sequences
}

export function logUri(jobId: number, name: string): vscode.Uri {
  // filename ends in .log so users get log syntax highlighting if installed
  const safe = name.replace(/[^\w.-]+/g, "_");
  return vscode.Uri.parse(`${SCHEME}:/${jobId}-${safe}.log`);
}

function jobIdFromUri(uri: vscode.Uri): number {
  return Number(uri.path.replace(/^\//, "").split("-")[0]);
}

function isFinal(jobState: string): boolean {
  return jobState !== "" && jobState !== "QUEUED" && jobState !== "RUNNING";
}

export class LogTailer implements vscode.TextDocumentContentProvider {
  private _onDidChange = new vscode.EventEmitter<vscode.Uri>();
  readonly onDidChange = this._onDidChange.event;

  private tails = new Map<string, Tail>();
  private timer: NodeJS.Timeout | undefined;
  private reopenListener: vscode.Disposable;

  constructor(private api: GqApi) {
    // a closed log tab reopened via "Reopen Closed Editor" / recent files
    // bypasses open(); re-create its tail so it resumes updating
    this.reopenListener = vscode.workspace.onDidOpenTextDocument((doc) => {
      if (doc.uri.scheme === SCHEME && !this.tails.has(doc.uri.toString())) {
        this.track(doc.uri, jobIdFromUri(doc.uri));
      }
    });
  }

  provideTextDocumentContent(uri: vscode.Uri): string {
    return this.tails.get(uri.toString())?.text ?? "loading…";
  }

  async open(jobId: number, name: string): Promise<void> {
    const uri = logUri(jobId, name);
    const tail = this.tails.get(uri.toString()) ?? this.track(uri, jobId);
    tail.nextPollAt = 0; // wake an idle tail (e.g. job was requeued)
    const doc = await vscode.workspace.openTextDocument(uri);
    await vscode.window.showTextDocument(doc, { preview: true, preserveFocus: false });
    await this.poll(uri);
    // land at the end of the log once the first content is in
    setTimeout(() => this.jumpToEnd(uri), 150);
  }

  private track(uri: vscode.Uri, jobId: number): Tail {
    const tail: Tail = {
      jobId,
      text: "",
      offset: 0,
      jobState: "",
      polling: false,
      nextPollAt: 0,
      decoder: new TextDecoder("utf-8"),
    };
    this.tails.set(uri.toString(), tail);
    this.ensureTimer();
    return tail;
  }

  private ensureTimer(): void {
    if (this.timer) {
      return;
    }
    this.timer = setInterval(() => this.sweep(), 1000);
  }

  private sweep(): void {
    const openUris = new Set(
      vscode.workspace.textDocuments
        .filter((d) => d.uri.scheme === SCHEME)
        .map((d) => d.uri.toString())
    );
    for (const key of [...this.tails.keys()]) {
      if (!openUris.has(key)) {
        this.tails.delete(key); // tab closed
        continue;
      }
      const tail = this.tails.get(key)!;
      if (!tail.polling && Date.now() >= tail.nextPollAt) {
        void this.poll(vscode.Uri.parse(key));
      }
    }
    if (this.tails.size === 0 && this.timer) {
      clearInterval(this.timer);
      this.timer = undefined;
    }
  }

  private async poll(uri: vscode.Uri): Promise<void> {
    const tail = this.tails.get(uri.toString());
    if (!tail || tail.polling) {
      return; // a fetch for this tail is already in flight
    }
    tail.polling = true;
    try {
      let gotData = false;
      for (let i = 0; i < MAX_CHUNKS_PER_POLL; i++) {
        const chunk = await this.api.logs(tail.jobId, tail.offset);
        tail.jobState = chunk.jobState;
        if (chunk.bytes.byteLength === 0) {
          break;
        }
        tail.text += tail.decoder.decode(chunk.bytes, { stream: true });
        tail.offset = chunk.nextOffset;
        gotData = true;
      }
      if (gotData) {
        this._onDidChange.fire(uri);
        // the provider is re-queried async; scroll after the doc updates
        setTimeout(() => this.autoScroll(uri), 100);
      }
      tail.nextPollAt = Date.now() + (isFinal(tail.jobState) ? IDLE_POLL_MS : 0);
    } catch {
      // daemon briefly unreachable: keep what we have, back off a little
      tail.nextPollAt = Date.now() + ERROR_RETRY_MS;
    } finally {
      tail.polling = false;
    }
  }

  /** Follow new output only while the cursor sits at the bottom of the log. */
  private autoScroll(uri: vscode.Uri): void {
    for (const editor of this.editorsFor(uri)) {
      if (!editor.selection.isEmpty) {
        continue; // don't clobber a selection
      }
      if (editor.selection.active.line < editor.document.lineCount - 3) {
        continue; // user scrolled up to read; leave them alone
      }
      this.reveal(editor);
    }
  }

  private jumpToEnd(uri: vscode.Uri): void {
    for (const editor of this.editorsFor(uri)) {
      const at = editor.selection.active;
      if (editor.selection.isEmpty && at.line === 0 && at.character === 0) {
        this.reveal(editor); // untouched, freshly opened editor
      }
    }
  }

  private reveal(editor: vscode.TextEditor): void {
    const pos = new vscode.Position(editor.document.lineCount - 1, 0);
    editor.selection = new vscode.Selection(pos, pos);
    editor.revealRange(new vscode.Range(pos, pos));
  }

  private editorsFor(uri: vscode.Uri): vscode.TextEditor[] {
    return vscode.window.visibleTextEditors.filter(
      (e) => e.document.uri.toString() === uri.toString()
    );
  }

  dispose(): void {
    if (this.timer) {
      clearInterval(this.timer);
    }
    this.reopenListener.dispose();
    this._onDidChange.dispose();
  }
}
