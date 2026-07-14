# gpu-queue

A mini-Slurm for a single shared GPU machine (chai). Submit jobs from the
terminal; a daemon runs them as GPUs become free; a VSCode extension shows
the queue and live logs.

**Why:** replace the "ask Claude to run jobs and poll them" workflow. Jobs are
submitted with `gq submit`, queue up, grab GPUs when available, and their
output is one click away in VSCode (via Remote-SSH — everything runs on chai,
so the extension talks to the daemon over localhost).

## Locked-in design decisions

- **Single machine.** No SSH dispatch. Install the tool per-machine if needed.
- **GPU request = count only** (`--gpus N`). No VRAM matching, no fractional
  sharing. GPUs on the box are treated as interchangeable.
- **Availability = queue ledger + live NVML check.** chai is shared with other
  users who do NOT use this tool. A GPU is dispatchable only if (a) the
  daemon's own allocation ledger says it's free AND (b) NVML shows no external
  compute process on it (VRAM/util below threshold). Externally-busy GPUs are
  off-limits and re-checked every scheduler cycle. No fairness/accounting —
  other users are outside the system, we just avoid stepping on them.
- **FIFO + backfill.** Head-of-queue order; if the head job needs more GPUs
  than are free, a later job that fits may run. Since jobs have no time
  limits, backfill can starve a multi-GPU head job — provide a hold/drain
  mechanism (e.g. `gq hold-gpus` or automatic reservation) so freed GPUs can
  accumulate for it.
- **Job spec: both forms from day one.**
  - Inline: `gq submit --gpus 1 --name foo -- python train.py --lr 1e-4`
  - Script: `gq submit job.sh` with `#GQ gpus=2`-style directives (sbatch-like)
  - Submit captures cwd and conda env (user works in conda, e.g.
    `fm_sudoku_new`) so the job runs exactly as it would interactively.
- **Stack:** Python daemon + CLI (one package, SQLite state, small localhost
  HTTP API, pynvml). Extension in TypeScript.
- **Extension v1 scope:** job tree (running/queued/finished + GPU + runtime),
  click job → live-tailing log, cancel/requeue from the UI. Submit-from-UI is
  explicitly deferred — the CLI covers it.

## Architecture

```
gpu_queue/            Python package (daemon + CLI, installed as `gq` / `gqd`)
├── daemon.py         scheduler loop, dispatch, reaping, recovery
├── scheduler.py      FIFO + backfill; GPU availability (ledger ∧ NVML)
├── db.py             SQLite job store (~/.gpu-queue/db.sqlite)
├── runner.py         process spawn: setsid group, CUDA_VISIBLE_DEVICES,
│                     cwd/env restore, stdout+stderr → ~/.gpu-queue/logs/<id>.log
├── api.py            localhost HTTP API: submit/list/get/logs/cancel/requeue/gpus
├── cli.py            gq submit|ls|logs -f|cancel|requeue|gpus|daemon start/stop/status
└── jobscript.py      #GQ directive parser
extension/            VSCode extension (TypeScript): tree view, log tail,
                      cancel/requeue commands, status bar counts; polls the API
docs/
```

Job lifecycle: `QUEUED → RUNNING → DONE | FAILED | CANCELLED`. Store per job:
command, workdir, env snapshot, gpus_requested, assigned gpu ids, pid,
timestamps, exit code, log path.

Robustness requirements:
- Cancel kills the whole process tree (process group), not just the parent.
- On daemon restart, reconcile DB vs live pids (jobs survive daemon crashes).
- Daemon runs as a systemd **user** service; `gq daemon start` sets it up.
- Two jobs must never be assigned the same GPU (ledger is authoritative;
  NVML check is advisory-only for *external* usage — don't build racy
  nvidia-smi-only allocation).

## Build order

1. **Core** — daemon + CLI: inline submit, strict FIFO, ledger+NVML
   availability, log capture, `gq ls/logs/cancel`. Already usable day-to-day.
2. **Queue features** — script-file submit, backfill (+ starvation guard),
   requeue, systemd unit, conda-env capture polish.
3. **VSCode extension** — tree view, live log tail, cancel/requeue, status bar.
4. **Polish** — completion notifications, `gq gpus` view, docs, packaging.
