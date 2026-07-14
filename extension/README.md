# gpu-queue VSCode extension

Shows the gpu-queue daemon's queue in a tree view (Running / Queued /
Finished), live-tails job logs, and lets you cancel/requeue jobs from the UI.
A status bar item shows running/queued counts and free GPUs.

Everything runs on the GPU machine: connect with **Remote-SSH** and install
the extension on the remote, where it talks to the daemon over localhost.

## Build & install

```bash
cd extension
npm install
npm run compile
npm run package                 # -> gpu-queue-0.1.0.vsix
code --install-extension gpu-queue-0.1.0.vsix   # in the Remote-SSH window
```

## Use

- **GPU Queue** icon in the activity bar → job tree. Click a job to open its
  live-tailing log. Inline buttons: cancel (queued/running), requeue (finished).
- Status bar: `gq: 1▶ 2⏳ · 3/4 free` — click to focus the tree.
- Notifications fire when a running job finishes or fails (configurable).

## Settings

| Setting                      | Default | Meaning                          |
|------------------------------|---------|----------------------------------|
| `gpuQueue.port`              | 47563   | daemon API port (match `GQ_PORT`)|
| `gpuQueue.pollIntervalMs`    | 2000    | queue/GPU poll interval          |
| `gpuQueue.finishedLimit`     | 15      | finished jobs shown in the tree  |
| `gpuQueue.notifyOnCompletion`| true    | toast when a job finishes/fails  |

Submit-from-UI is deliberately out of scope — use `gq submit`.
