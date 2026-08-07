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
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Sequence, Set, Tuple

from .db import Job, Tenant

# External compute processes using less VRAM than this are ignored (Xorg etc.)
EXT_MEM_MB = int(os.environ.get("GQ_EXT_MEM_MB", "256"))
RESERVE_AFTER_S = float(os.environ.get("GQ_RESERVE_AFTER_S", "300"))

# Per-GPU capacity held back: ~600MB is reserved by the driver on an A6000,
# plus slack so a card is never packed to its last megabyte.
VRAM_HEADROOM_MB = int(os.environ.get("GQ_VRAM_HEADROOM_MB", "1024"))
# Charged per tenant on top of its declared budget. --vram is what the process
# should occupy in nvidia-smi (context included), so this is variance margin,
# not a full CUDA-context allowance.
VRAM_OVERHEAD_MB = int(os.environ.get("GQ_VRAM_OVERHEAD_MB", "256"))

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
    `fake_procs` lets a test pretend one of our own pids is using VRAM, which
    is the only way to exercise the policing path without a real GPU. It is
    keyed by GPU index, like NVML: a process shows up only on the card it is
    actually using.
    """

    def __init__(self, count: int):
        self.count = count
        # gpu index -> {pid: VRAM MB}
        self.fake_procs: Dict[int, Dict[int, Optional[int]]] = {}

    def snapshot(self) -> List[GpuInfo]:
        busy = {
            int(x)
            for x in os.environ.get("GQ_FAKE_BUSY", "").split(",")
            if x.strip()
        }
        out = []
        for i in range(self.count):
            procs: Dict[int, Optional[int]] = dict(self.fake_procs.get(i, {}))
            if i in busy:
                procs[99999 + i] = 8000
            out.append(
                GpuInfo(
                    index=i,
                    name="FakeGPU",
                    mem_used_mb=8000 if i in busy else 10,
                    mem_total_mb=24000,
                    util_pct=90 if i in busy else 0,
                    compute_procs=procs,
                )
            )
        return out


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
class VramViolation:
    job: Job
    used_mb: int
    budget_mb: int


def vram_violations(
    gpus: Sequence[GpuInfo], running_jobs: Sequence[Job]
) -> List[VramViolation]:
    """Share jobs using more VRAM than they declared.

    Declared budgets are an honour system -- CUDA has no per-process cap we can
    impose from outside -- and the failure mode is nasty: CUDA hands out memory
    first-come-first-served, so an over-running job usually survives while its
    correctly-declared neighbour takes the OOM. Policing turns that into a
    failure for the job that actually caused it.

    Only share jobs are checked; an exclusive job owns its card and can use all
    of it. Fails open: NVML reports per-process memory as None under some driver
    configurations, and a job we cannot measure is never killed.
    """
    by_pgid = {j.pid: j for j in running_jobs if j.pid and j.vram_mb is not None}
    if not by_pgid:
        return []

    used: Dict[int, int] = {}
    unmeasurable: Set[int] = set()
    for gpu in gpus:
        for pid, mem_mb in gpu.compute_procs.items():
            pgid = _pgid_of(pid)
            job = by_pgid.get(pgid)
            if job is None:
                continue
            if mem_mb is None:
                unmeasurable.add(job.id)
                continue
            used[job.id] = used.get(job.id, 0) + mem_mb

    out = []
    for job in by_pgid.values():
        if job.id in unmeasurable or job.id not in used:
            continue
        # The scheduler reserved budget + overhead for this job, so that is the
        # point at which it starts eating into someone else's reservation.
        budget = job.vram_mb + VRAM_OVERHEAD_MB
        if used[job.id] > budget:
            out.append(VramViolation(job, used[job.id], job.vram_mb))
    return out


@dataclass
class Assignment:
    job: Job
    gpu_ids: List[int]


@dataclass
class GpuSlot:
    """A dispatchable GPU and what is left of it.

    `free_mb` is capacity for *share* jobs (total minus headroom minus what
    current tenants reserved); `empty` means no gpu-queue job is on it at all,
    which is what an exclusive job requires.
    """

    index: int
    free_mb: int
    empty: bool


class Scheduler:
    def __init__(self):
        # job id -> monotonic time when it first became a blocked head job
        self._head_blocked_since: Dict[int, float] = {}

    def _place(self, job: Job, avail: List[GpuSlot]) -> Optional[List[int]]:
        """GPU indices for one job, or None if it doesn't fit right now.

        Mutates `avail`: an exclusive job takes whole slots out of the pool, a
        share job just draws down one slot's capacity. Returning [] (a 0-GPU
        job) is success, so callers must test `is not None`, not truthiness.
        """
        if job.vram_mb is None:
            # Exclusive: whole GPUs, and only ones nothing else is sharing.
            empty = [s for s in avail if s.empty]
            if len(empty) < job.gpus_requested:
                return None
            chosen = empty[: job.gpus_requested]
            for slot in chosen:
                avail.remove(slot)
            return [slot.index for slot in chosen]

        # Share: one GPU with room to spare. Best-fit — the tightest slot that
        # works — so empty cards stay empty for exclusive jobs as long as
        # possible instead of every card picking up one small tenant.
        need = job.vram_mb + VRAM_OVERHEAD_MB
        candidates = [s for s in avail if s.free_mb >= need]
        if not candidates:
            return None
        slot = min(candidates, key=lambda s: s.free_mb)
        slot.free_mb -= need
        slot.empty = False
        return [slot.index]

    def plan(
        self,
        queued: Sequence[Job],
        free_gpus: Sequence[GpuSlot],
        now: Optional[float] = None,
    ) -> List[Assignment]:
        """FIFO with backfill over the queued jobs and dispatchable GPUs.

        Backfill: if the head job doesn't fit, later jobs that do fit may run —
        unless the head has been blocked past RESERVE_AFTER_S, in which case
        the remaining capacity is reserved so it accumulates for it.
        """
        if now is None:
            now = time.monotonic()
        # Copy the slots, not just the list: _place draws down free_mb, and
        # plan() must stay a pure decision function the caller can re-run.
        avail = [replace(s) for s in free_gpus]
        out: List[Assignment] = []
        head_blocked = False

        for job in queued:
            gpu_ids = self._place(job, avail)
            if gpu_ids is not None:
                out.append(Assignment(job, gpu_ids))
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
        ledger: Dict[int, List[Tenant]],
        running_jobs: Sequence[Job],
    ) -> List[GpuSlot]:
        """Dispatchable GPUs and their remaining capacity.

        A GPU is dispatchable if no external process is on it and no exclusive
        job of ours holds it. Capacity is charged against tenants' *declared*
        budgets, never live NVML readings — a job that hasn't allocated yet
        would otherwise look free and the GPU would be overcommitted.
        """
        ours = _our_pgids(running_jobs)

        slots: List[GpuSlot] = []
        for g in gpus:
            if externally_busy(g, ours):
                continue  # another user's GPU: off-limits entirely

            tenants = ledger.get(g.index, [])
            if any(t.vram_mb is None for t in tenants):
                # An exclusive job declared no budget, so there is no number to
                # subtract: the card can't be shared with it.
                continue

            reserved = sum(t.vram_mb + VRAM_OVERHEAD_MB for t in tenants)
            slots.append(
                GpuSlot(
                    index=g.index,
                    free_mb=g.mem_total_mb - VRAM_HEADROOM_MB - reserved,
                    empty=not tenants,
                )
            )
        return slots

