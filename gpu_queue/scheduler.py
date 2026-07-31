"""FIFO + backfill scheduling, and GPU availability (ledger ∧ NVML).

chai is shared with people who don't use gpu-queue, so a GPU is dispatchable
only if our own allocation ledger says it's free AND NVML shows no external
compute process on it. The ledger is authoritative for our jobs; the NVML
check is advisory-only, for staying off other users' GPUs.

Starvation guard: backfill is on, but once the head-of-queue job has been
blocked for GQ_RESERVE_AFTER_S seconds (default 300), all free GPUs are
reserved for it so they can accumulate instead of being backfilled forever.
"""

import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .db import Job

# External compute processes using less VRAM than this are ignored (Xorg etc.)
EXT_MEM_MB = int(os.environ.get("GQ_EXT_MEM_MB", "256"))
RESERVE_AFTER_S = float(os.environ.get("GQ_RESERVE_AFTER_S", "300"))


@dataclass
class GpuInfo:
    index: int
    name: str
    mem_used_mb: int
    mem_total_mb: int
    util_pct: int
    # pids of compute processes with their VRAM usage in MB (None if unknown)
    compute_procs: Dict[int, Optional[int]] = field(default_factory=dict)


class GpuMonitor:
    """Interface: snapshot() returns current GpuInfo for every GPU."""

    def snapshot(self) -> List[GpuInfo]:  # pragma: no cover - interface
        raise NotImplementedError


class NvmlMonitor(GpuMonitor):
    def __init__(self):
        import pynvml

        self._nvml = pynvml
        pynvml.nvmlInit()
        self._count = pynvml.nvmlDeviceGetCount()

    def _mem_used_total_mb(self, h) -> Tuple[int, int]:
        """VRAM (used, total) in MB, excluding driver-reserved memory.

        NVML's v1 struct folds ~600MB of reserved memory into `used`, which
        reads as 13% utilisation on an idle GPU. v2 splits it out; fall back
        to v1 on drivers that lack it.
        """
        nvml = self._nvml
        try:
            m = nvml.nvmlDeviceGetMemoryInfo(h, version=nvml.nvmlMemory_v2)
        except (nvml.NVMLError, AttributeError, TypeError):
            m = nvml.nvmlDeviceGetMemoryInfo(h)
        return m.used // (1024 * 1024), m.total // (1024 * 1024)

    def snapshot(self) -> List[GpuInfo]:
        nvml = self._nvml
        gpus = []
        for i in range(self._count):
            h = nvml.nvmlDeviceGetHandleByIndex(i)
            mem_used_mb, mem_total_mb = self._mem_used_total_mb(h)
            try:
                util = nvml.nvmlDeviceGetUtilizationRates(h).gpu
            except nvml.NVMLError:
                util = 0
            procs: Dict[int, Optional[int]] = {}
            try:
                plist = list(nvml.nvmlDeviceGetComputeRunningProcesses(h))
            except nvml.NVMLError:
                plist = []
            for p in plist:
                used = p.usedGpuMemory
                procs[p.pid] = None if used is None else used // (1024 * 1024)
            name = nvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            gpus.append(
                GpuInfo(
                    index=i,
                    name=name,
                    mem_used_mb=mem_used_mb,
                    mem_total_mb=mem_total_mb,
                    util_pct=util,
                    compute_procs=procs,
                )
            )
        return gpus


class FakeMonitor(GpuMonitor):
    """GQ_FAKE_GPUS=N replaces NVML — for dev boxes and tests.

    GQ_FAKE_BUSY="1,3" marks those indices as externally busy.
    """

    def __init__(self, count: int):
        self.count = count

    def snapshot(self) -> List[GpuInfo]:
        busy = {
            int(x)
            for x in os.environ.get("GQ_FAKE_BUSY", "").split(",")
            if x.strip()
        }
        return [
            GpuInfo(
                index=i,
                name="FakeGPU",
                mem_used_mb=8000 if i in busy else 10,
                mem_total_mb=24000,
                util_pct=90 if i in busy else 0,
                compute_procs={99999 + i: 8000} if i in busy else {},
            )
            for i in range(self.count)
        ]


def make_monitor() -> GpuMonitor:
    fake = os.environ.get("GQ_FAKE_GPUS")
    if fake:
        return FakeMonitor(int(fake))
    return NvmlMonitor()


def _our_pgids(running_jobs: Sequence[Job]) -> Set[int]:
    pgids = set()
    for job in running_jobs:
        if job.pid:
            pgids.add(job.pid)  # we setsid, so pgid == job pid
    return pgids


def _pgid_of(pid: int) -> Optional[int]:
    try:
        return os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        return None


def externally_busy(gpu: GpuInfo, our_pgids: Set[int]) -> bool:
    """True if someone outside gpu-queue is using this GPU."""
    external_found = False
    for pid, mem_mb in gpu.compute_procs.items():
        pgid = _pgid_of(pid)
        if pgid is not None and pgid in our_pgids:
            continue
        if mem_mb is not None and mem_mb < EXT_MEM_MB:
            continue  # tiny contexts (display servers etc.)
        external_found = True
    return external_found


@dataclass
class Assignment:
    job: Job
    gpu_ids: List[int]


class Scheduler:
    def __init__(self):
        # job id -> monotonic time when it first became a blocked head job
        self._head_blocked_since: Dict[int, float] = {}

    def plan(
        self,
        queued: Sequence[Job],
        free_gpus: Sequence[int],
        now: Optional[float] = None,
    ) -> List[Assignment]:
        """FIFO with backfill over the queued jobs and free GPU indices.

        Backfill: if the head job doesn't fit, later jobs that do fit may run —
        unless the head has been blocked past RESERVE_AFTER_S, in which case
        the remaining free GPUs are reserved so they accumulate for it.
        """
        if now is None:
            now = time.monotonic()
        avail = list(free_gpus)
        out: List[Assignment] = []
        head_blocked = False

        for job in queued:
            fits = job.gpus_requested <= len(avail)
            if fits:
                out.append(Assignment(job, avail[: job.gpus_requested]))
                avail = avail[job.gpus_requested :]
                self._head_blocked_since.pop(job.id, None)
                continue
            if head_blocked:
                continue  # only the first blocked job gets reservation rights
            head_blocked = True
            blocked_since = self._head_blocked_since.setdefault(job.id, now)
            if now - blocked_since >= RESERVE_AFTER_S:
                avail = []  # reserve everything free for the starving head

        # drop tracking for jobs no longer at a blocked position
        queued_ids = {j.id for j in queued}
        scheduled_ids = {a.job.id for a in out}
        for jid in list(self._head_blocked_since):
            if jid not in queued_ids or jid in scheduled_ids:
                del self._head_blocked_since[jid]
        return out

    def free_gpus(
        self,
        gpus: Sequence[GpuInfo],
        ledger: Dict[int, int],
        running_jobs: Sequence[Job],
    ) -> List[int]:
        """GPU indices that are free per our ledger AND not externally busy."""
        ours = _our_pgids(running_jobs)
        return [
            g.index
            for g in gpus
            if g.index not in ledger and not externally_busy(g, ours)
        ]
