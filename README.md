# gpu-queue

A mini-Slurm for a single shared GPU machine. Submit jobs with `gq submit`;
a daemon runs them as GPUs become free; a VSCode extension shows the queue
and live logs.

```bash
gq submit --gpus 1 --name lr-sweep -- python train.py --lr 1e-4
gq ls
gq logs 42 -f
gq cancel 42
```

See `CLAUDE.md` for the full design and build plan.
