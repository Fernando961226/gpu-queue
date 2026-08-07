# gpu-queue

A mini-Slurm for a single shared GPU machine. Submit jobs with `gq submit`;
a daemon runs them as GPUs become free; a VSCode extension shows the queue
and live logs.

```bash
gq submit --gpus 1 --name lr-sweep -- python train.py --lr 1e-4
gq submit --vram 8G -- python train.py     # share a GPU with other small jobs
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
- **Survives logout.** `gq daemon start` enables systemd linger, so the daemon
  keeps dispatching after you disconnect and comes back on reboot. Without it
  the user manager — and the queue — stops with your last SSH session. If
  polkit refuses, `gq` says so and you finish with
  `sudo loginctl enable-linger $USER`.

State lives in `~/.gpu-queue/` (SQLite DB + per-job logs).

### Sharing a GPU

A 48GB card reserved for a 4GB job is mostly wasted. `--vram` declares a budget
instead of claiming the whole GPU, and jobs that fit together run together:

```bash
gq submit --vram 8G --name a -- python train.py     # both land on the same GPU
gq submit --vram 8G --name b -- python train.py
gq gpus                                             # shows reserved/capacity + tenants
```

**`--vram` is what `nvidia-smi` shows for your process**, CUDA context included
— not what your tensors weigh. Measure a run on its own, then declare that plus
~20%.

Budgets are enforced. There is no way to cap a process's VRAM from outside, and
CUDA hands out memory first-come-first-served, so an over-running job would
usually survive while its correctly-declared neighbour takes the OOM. Instead
the daemon watches per-process VRAM and evicts the job that went over its own
declaration: three consecutive cycles over budget → SIGTERM → `FAILED` with a
note telling you what to resubmit with. A job whose memory NVML cannot read is
never killed.

Exclusive jobs (`--gpus N`) and share jobs never mix on a card, so a `--gpus 1`
job still gets a whole GPU to itself.

### Job scripts

```bash
#!/bin/bash
#GQ gpus=2 name=big-sweep
python train.py --lr 1e-4
```

```bash
#!/bin/bash
#GQ vram=8G name=small-sweep
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
| `GQ_VRAM_HEADROOM_MB` | `1024`       | per-GPU VRAM held back from sharing       |
| `GQ_VRAM_OVERHEAD_MB` | `256`        | charged per share job on top of its budget |
| `GQ_VRAM_STRIKES`   | `3`            | cycles over budget before a job is evicted |
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
