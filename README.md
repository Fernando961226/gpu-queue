# gpu-queue

A mini-Slurm for a single shared GPU machine. Submit jobs with `gq submit`;
a daemon runs them as GPUs become free; a VSCode extension shows the queue
and live logs.

```bash
gq submit --gpus 1 --name lr-sweep -- python train.py --lr 1e-4
gq submit job.sh          # sbatch-style: #GQ gpus=2 directives in the script
gq ls
gq logs 42 -f
gq cancel 42
gq requeue 42
gq gpus
```

## Install

```bash
curl -fsSL https://raw.githubusercontent.com/Fernando961226/gpu-queue/main/install.sh | bash
gq daemon start           # installs + starts a systemd user service
```

No prerequisites beyond `python3` and `curl` — nothing to install first, and
nothing added to your conda envs. The installer builds a private venv at
`~/.local/share/gpu-queue/venv` (preferring the *system* python, so the daemon
doesn't depend on conda) and symlinks `gq`/`gqd` into `~/.local/bin`.

| Command | Effect |
|---------|--------|
| `install.sh` | install, or upgrade in place if already installed |
| `install.sh --with-daemon` | also enable + start the systemd user service |
| `install.sh --dev` | editable install + test deps, from a local checkout |
| `install.sh --uninstall` | remove the venv, symlinks, and service (keeps `~/.gpu-queue`) |

Overridable via `GQ_PREFIX`, `GQ_BIN_DIR`, `GQ_PYTHON`. If `uv` happens to be
installed it's used for speed; it is never required.

## How it works

- **Submit captures your context.** The job runs with the cwd and full
  environment (including your active conda env) you submitted from, with
  `CUDA_VISIBLE_DEVICES` set to the assigned GPUs.
- **Shared-machine aware.** A GPU is used only if the daemon's own ledger
  says it's free *and* NVML shows no external compute process on it —
  other users' jobs are left alone.
- **FIFO + backfill with a starvation guard.** If the head job needs more
  GPUs than are free, smaller jobs may backfill; after the head has waited
  `GQ_RESERVE_AFTER_S` (default 300s), freed GPUs are reserved so they can
  accumulate for it.
- **Robust.** Cancel kills the whole process tree; jobs survive daemon
  restarts (the daemon reconciles the DB against live pids on startup).

State lives in `~/.gpu-queue/` (SQLite DB + per-job logs).

### Job scripts

```bash
#!/bin/bash
#GQ gpus=2 name=big-sweep
python train.py --lr 1e-4
```

### Configuration (env vars)

| Variable            | Default        | Meaning                                   |
|---------------------|----------------|-------------------------------------------|
| `GQ_HOME`           | `~/.gpu-queue` | state directory                           |
| `GQ_PORT`           | `47563`        | localhost API port                        |
| `GQ_POLL_S`         | `2`            | scheduler cycle interval                  |
| `GQ_RESERVE_AFTER_S`| `300`          | head-job wait before GPUs are reserved    |
| `GQ_EXT_MEM_MB`     | `256`          | VRAM below which external procs are ignored |
| `GQ_FAKE_GPUS`      | unset          | fake N GPUs instead of NVML (dev/testing) |

## Development

```bash
git clone https://github.com/Fernando961226/gpu-queue.git && cd gpu-queue
./install.sh --dev                   # editable venv + pytest
~/.local/share/gpu-queue/venv/bin/python -m pytest tests/
GQ_FAKE_GPUS=4 GQ_HOME=/tmp/gq gqd   # daemon with fake GPUs, foreground
```

See `CLAUDE.md` for the full design and build plan. The VSCode extension
(job tree + live log tail) lives in `extension/`.
