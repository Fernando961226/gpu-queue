# How gpu-queue works

Deeper detail than the README; the design contract lives in `CLAUDE.md`.

## Components

- **`gqd`** — the daemon. Owns the SQLite DB (`~/.gpu-queue/db.sqlite`),
  runs the scheduler loop every `GQ_POLL_S` (2s), and serves the localhost
  HTTP API. Runs as a systemd *user* service (`gpu-queue.service`).
- **`gq`** — the CLI. Talks only to the API; never touches the DB.
- **VSCode extension** — also talks only to the API (over localhost, via
  Remote-SSH on the GPU machine).

## Job lifecycle

```
QUEUED ──dispatch──> RUNNING ──exit 0──────> DONE
                        │ ────exit != 0────> FAILED
                        │ ────gq cancel────> CANCELLED
```

Submit captures the caller's **cwd and full environment** (that's what makes
conda envs "just work" — the env snapshot includes the activated env's PATH).
At dispatch the job is spawned with that environment plus
`CUDA_VISIBLE_DEVICES=<assigned ids>`, `GQ_JOB_ID`, and `GQ_EXIT_FILE`.

### The process tree

Each job runs via a bash wrapper in **its own session** (`setsid`), so
pgid == leader pid. Consequences:

- Cancel = `killpg(SIGTERM)`, escalating to `SIGKILL` after
  `GQ_CANCEL_GRACE_S` (10s) if anything in the group survives.
- A job counts as *finished* only when its **whole process group** is gone,
  not when the wrapper exits. If the script exits leaving background
  children, the daemon SIGKILLs the stragglers (slurm-style) before
  finalizing.
- The wrapper writes the exit code to `~/.gpu-queue/logs/<id>.exit` before
  exiting. That file is authoritative: it lets a restarted daemon finalize
  jobs it can no longer `waitpid` (and defeats pid-reuse confusion).

### Daemon restarts

On startup the daemon reconciles DB vs reality: RUNNING jobs whose process
group is alive (and have no exit file) are re-adopted and tracked by
pid + exit file; the rest are finalized from their exit files. The systemd
unit uses `KillMode=process`, so `systemctl --user stop/restart gpu-queue`
signals only the daemon — jobs keep running.

### Logout and linger

A systemd *user* manager normally exits with your last login session, which
for a queue means dispatching silently stops the moment you close SSH, and
nothing restarts after a reboot until someone logs in. `gq daemon start`
therefore calls `loginctl enable-linger` for the current user. It's best
effort: if polkit denies it, `gq` prints the `sudo` command rather than
failing, since the daemon still works fine while you're logged in.

## Scheduling

Every cycle:

1. **Snapshot GPUs** via NVML (or `GQ_FAKE_GPUS` fakes).
2. **Reap** finished/cancelled jobs (frees their ledger entries).
3. **Compute free GPUs** = not in the ledger (our RUNNING jobs' assignments)
   ∧ no *external* compute process on them. External = an NVML compute pid
   whose process group isn't one of our jobs, using ≥ `GQ_EXT_MEM_MB` VRAM.
   Other users' GPUs are simply off-limits; they're re-checked every cycle.
4. **Plan** FIFO with backfill: walk the queue in submit order, assign free
   GPUs. If the head doesn't fit, later jobs that fit may run — until the
   head has been blocked ≥ `GQ_RESERVE_AFTER_S` (300s), after which all free
   GPUs are reserved for it so they can accumulate (starvation guard). Only
   the first blocked job gets reservation rights.

The ledger is authoritative for our own allocations — two gpu-queue jobs can
never share a GPU. The NVML check is advisory and only guards against
*external* users; it is not used for our own accounting (no racy
nvidia-smi-based allocation).

Jobs requesting more GPUs than the machine has are rejected at submit.

## HTTP API

Bound to `127.0.0.1:GQ_PORT` (47563). No auth — localhost only, single-user
by design.

| Method | Path                    | Notes                                    |
|--------|-------------------------|------------------------------------------|
| GET    | `/api/health`           | `{ok, version}`                          |
| GET    | `/api/jobs`             | `?state=QUEUED,RUNNING&limit=N`, newest first |
| GET    | `/api/jobs/<id>`        | full job record                          |
| GET    | `/api/jobs/<id>/logs`   | `?offset=N`, text/plain, ≤512KB chunks; `X-Log-Offset` = next offset, `X-Job-State` for tail termination |
| GET    | `/api/gpus`             | per-GPU: state free/allocated/external, mem, util, job id |
| POST   | `/api/submit`           | `{name?, command[], workdir, env{}, gpus, conda_env?}` |
| POST   | `/api/jobs/<id>/cancel` | queued → CANCELLED; running → SIGTERM + grace |
| POST   | `/api/jobs/<id>/requeue`| finished jobs only; same id, back of queue |
| POST   | `/api/shutdown`         | stop the daemon (jobs keep running)      |

## Files

```
~/.gpu-queue/
├── db.sqlite        job store (WAL)
└── logs/
    ├── <id>.log     stdout+stderr, header per (re)start, appended on requeue
    └── <id>.exit    exit code sentinel written by the job wrapper
```
