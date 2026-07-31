"""Scratch job for trying out gq by hand — not a pytest test.

Prints the captured context (cwd, conda env, assigned GPUs), then hammers
every visible GPU with large matmuls so you can watch utilisation climb in
`gq gpus` / `nvidia-smi`.

Needs a torch build with CUDA, so submit it from an env that has one:

    conda activate fm_sudoku_new
    gq submit --gpus 1 --name stress -- python tests/dummy.py

    # bigger/longer, and on two GPUs:
    gq submit --gpus 2 --name stress2 -- python tests/dummy.py --size 20000 --seconds 120
"""
import argparse
import os
import socket
import time

p = argparse.ArgumentParser()
p.add_argument("--size", type=int, default=16384, help="square matrix edge length")
p.add_argument("--seconds", type=int, default=60, help="how long to keep the GPUs busy")
args = p.parse_args()

print("host:", socket.gethostname())
print("pid:", os.getpid())
print("cwd:", os.getcwd())
print("conda env:", os.environ.get("CONDA_DEFAULT_ENV"))
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
print("GQ_JOB_ID:", os.environ.get("GQ_JOB_ID"), flush=True)

import torch  # noqa: E402  (imported late so the context above prints even if torch is missing)

if not torch.cuda.is_available():
    raise SystemExit(
        "torch reports no CUDA device — submit from an env with a CUDA build, "
        "e.g. `conda activate fm_sudoku_new` before `gq submit`"
    )

# CUDA renumbers the assigned GPUs to 0..N-1, so this covers exactly what gq
# handed us, whichever physical GPUs those are.
devices = list(range(torch.cuda.device_count()))
print(f"torch {torch.__version__}, visible devices: {devices}", flush=True)
for d in devices:
    print(f"  cuda:{d} -> {torch.cuda.get_device_name(d)}", flush=True)

n = args.size
mats = []
for d in devices:
    a = torch.randn(n, n, device=f"cuda:{d}")
    b = torch.randn(n, n, device=f"cuda:{d}")
    mats.append((a, b))
    gib = torch.cuda.memory_allocated(d) / 1024**3
    print(f"  cuda:{d} allocated {gib:.2f} GiB for {n}x{n} matmuls", flush=True)

flop_per_matmul = 2 * n**3
deadline = time.time() + args.seconds
step = 0
print(f"stressing {len(devices)} GPU(s) for {args.seconds}s ...", flush=True)

while time.time() < deadline:
    step += 1
    t0 = time.time()
    for d, (a, b) in zip(devices, mats):
        c = a @ b            # noqa: F841 — result kept only to force the work
    for d in devices:
        torch.cuda.synchronize(d)
    dt = time.time() - t0
    tflops = (flop_per_matmul * len(devices)) / dt / 1e12
    peak = max(torch.cuda.max_memory_allocated(d) for d in devices) / 1024**3
    print(
        f"step {step:4d}  {dt:6.3f}s  {tflops:7.2f} TFLOP/s  peak_vram {peak:.2f} GiB",
        flush=True,
    )

print("done")
