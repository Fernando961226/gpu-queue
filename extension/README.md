# gpu-queue VSCode extension

Shows the gpu-queue daemon's queue in a tree view (Running / Queued /
Finished), live-tails job logs, and lets you cancel/requeue jobs from the UI.
A **GPUs** view above it shows per-GPU utilisation, VRAM and the owning job.
A status bar item shows running/queued counts and free GPUs.

Everything runs on the GPU machine: connect with **Remote-SSH** and install
the extension on the remote, where it talks to the daemon over localhost.

## Build & install

```bash
cd extension
npm install
npm run compile
npm run package                 # -> gpu-queue-0.2.0.vsix
code --install-extension gpu-queue-0.2.0.vsix   # in the Remote-SSH window
```

## Use

- **GPU Queue** icon in the activity bar → **GPUs** and **Jobs** views.
- GPUs: one row per GPU, `● GPU 0  ██████████ 100% · 3.3/48.0 GiB · #14 train`.
  Filled circle = running one of our jobs, hollow = free, orange warning =
  in use by someone outside gpu-queue (we never schedule there). Hover for the
  full device name and a utilisation sparkline of the last ~60s.
- Jobs: click a job to open its live-tailing log. Inline buttons: cancel
  (queued/running), requeue (finished).
- Status bar: `gq: 1▶ 2⏳ · 3/4 free` — click to focus the tree.
- Notifications fire when a running job finishes or fails (configurable).

## Settings

| Setting                      | Default | Meaning                          |
|------------------------------|---------|----------------------------------|
| `gpuQueue.port`              | 47563   | daemon API port (match `GQ_PORT`)|
| `gpuQueue.pollIntervalMs`    | 2000    | queue/GPU poll interval          |
| `gpuQueue.finishedLimit`     | 15      | finished jobs shown in the tree  |
| `gpuQueue.notifyOnCompletion`| true    | toast when a job finishes/fails  |
| `gpuQueue.gpuHistoryLength`  | 30      | util samples kept for the sparkline |

Submit-from-UI is deliberately out of scope — use `gq submit`.
