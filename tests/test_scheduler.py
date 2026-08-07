from dataclasses import dataclass
from typing import Optional

import gpu_queue.scheduler as sched
from gpu_queue.db import Tenant
from gpu_queue.scheduler import GpuInfo, GpuSlot, Scheduler, externally_busy

# Round numbers so the capacity arithmetic in these tests is readable.
GPU_MB = 48 * 1024


@dataclass
class FakeJob:
    id: int
    gpus_requested: int = 1
    vram_mb: Optional[int] = None  # None == exclusive job


def empty_slots(*indices):
    """Untouched GPUs: full capacity, available to exclusive jobs."""
    return [GpuSlot(i, GPU_MB - sched.VRAM_HEADROOM_MB, True) for i in indices]


def plan_ids(scheduler, jobs, free, now=0.0):
    return [(a.job.id, a.gpu_ids) for a in scheduler.plan(jobs, free, now=now)]


def test_fifo_order():
    s = Scheduler()
    jobs = [FakeJob(1), FakeJob(2), FakeJob(3)]
    assert plan_ids(s, jobs, empty_slots(0, 1)) == [(1, [0]), (2, [1])]


def test_no_double_assignment():
    s = Scheduler()
    jobs = [FakeJob(1, 2), FakeJob(2, 2)]
    out = plan_ids(s, jobs, empty_slots(0, 1, 2, 3))
    used = [g for _, gpus in out for g in gpus]
    assert len(used) == len(set(used)) == 4


def test_backfill_when_head_blocked():
    s = Scheduler()
    jobs = [FakeJob(1, 4), FakeJob(2, 1)]
    # head needs 4, only 2 free: job 2 backfills
    assert plan_ids(s, jobs, empty_slots(0, 1)) == [(2, [0])]


def test_only_first_blocked_job_reserves():
    s = Scheduler()
    jobs = [FakeJob(1, 4), FakeJob(2, 3), FakeJob(3, 1)]
    # head blocked, job 2 also blocked, job 3 backfills
    assert plan_ids(s, jobs, empty_slots(0, 1)) == [(3, [0])]


def test_reservation_stops_backfill_after_timeout():
    s = Scheduler()
    jobs = [FakeJob(1, 4), FakeJob(2, 1)]
    # first cycle at t=0: head becomes blocked, backfill allowed
    assert plan_ids(s, jobs, empty_slots(0, 1), now=0.0) == [(2, [0])]
    # much later: reservation kicks in, no more backfill
    later = sched.RESERVE_AFTER_S + 1
    assert plan_ids(s, jobs, empty_slots(0, 1), now=later) == []


def test_head_runs_when_enough_gpus_free():
    s = Scheduler()
    jobs = [FakeJob(1, 4), FakeJob(2, 1)]
    assert plan_ids(s, jobs, empty_slots(0, 1), now=0.0) == [(2, [0])]
    # head finally fits: it runs and its blocked-tracking is dropped
    out = plan_ids(s, jobs, empty_slots(0, 1, 2, 3, 4), now=sched.RESERVE_AFTER_S + 1)
    assert out == [(1, [0, 1, 2, 3]), (2, [4])]
    assert 1 not in s._head_blocked_since


def test_blocked_tracking_cleared_when_job_leaves_queue():
    s = Scheduler()
    plan_ids(s, [FakeJob(1, 4)], empty_slots(0), now=0.0)
    assert 1 in s._head_blocked_since
    plan_ids(s, [], empty_slots(0), now=1.0)  # job cancelled
    assert 1 not in s._head_blocked_since


def test_zero_gpu_job_runs_even_with_nothing_free():
    s = Scheduler()
    # [] is a successful placement, not a refusal — plan must test `is not None`
    assert plan_ids(s, [FakeJob(1, 0)], []) == [(1, [])]


# -- packing ---------------------------------------------------------------


def test_share_job_packs_onto_occupied_gpu():
    s = Scheduler()
    occupied = [GpuSlot(0, 20_000, False)]
    assert plan_ids(s, [FakeJob(1, 1, vram_mb=8192)], occupied) == [(1, [0])]


def test_share_job_rejected_when_capacity_short():
    s = Scheduler()
    # 8192 + overhead needed, only 8192 left
    tight = [GpuSlot(0, 8192, False)]
    assert plan_ids(s, [FakeJob(1, 1, vram_mb=8192)], tight) == []


def test_share_job_prefers_tightest_fit():
    s = Scheduler()
    slots = [GpuSlot(0, 40_000, False), GpuSlot(1, 12_000, False)]
    # Both fit; best-fit takes GPU 1 so the roomier card stays roomy.
    assert plan_ids(s, [FakeJob(1, 1, vram_mb=8192)], slots) == [(1, [1])]


def test_share_job_prefers_occupied_gpu_over_empty():
    s = Scheduler()
    slots = empty_slots(0) + [GpuSlot(1, 20_000, False)]
    # Keeps GPU 0 empty so an exclusive job can still have it.
    assert plan_ids(s, [FakeJob(1, 1, vram_mb=8192)], slots) == [(1, [1])]


def test_exclusive_job_refuses_shared_gpu():
    s = Scheduler()
    shared = [GpuSlot(0, 40_000, False)]  # plenty of room, but not empty
    assert plan_ids(s, [FakeJob(1, 1)], shared) == []


def test_exclusive_job_takes_empty_gpu_beside_a_shared_one():
    s = Scheduler()
    slots = [GpuSlot(0, 40_000, False)] + empty_slots(1)
    assert plan_ids(s, [FakeJob(1, 1)], slots) == [(1, [1])]


def test_two_share_jobs_do_not_double_book_capacity():
    s = Scheduler()
    # Room for one 20G job, not two: the second must wait.
    slots = [GpuSlot(0, 21_000, False)]
    jobs = [FakeJob(1, 1, vram_mb=20_480), FakeJob(2, 1, vram_mb=20_480)]
    assert plan_ids(s, jobs, slots) == [(1, [0])]


def test_two_share_jobs_fit_when_there_is_room():
    s = Scheduler()
    jobs = [FakeJob(1, 1, vram_mb=8192), FakeJob(2, 1, vram_mb=8192)]
    out = plan_ids(s, jobs, empty_slots(0))
    assert out == [(1, [0]), (2, [0])]  # both on the same GPU


def test_plan_does_not_mutate_callers_slots():
    s = Scheduler()
    slots = empty_slots(0)
    before = slots[0].free_mb
    plan_ids(s, [FakeJob(1, 1, vram_mb=8192)], slots)
    assert slots[0].free_mb == before and slots[0].empty is True


def test_share_job_blocked_head_still_reserves():
    s = Scheduler()
    # A share job too big for anything free becomes the blocked head; once it
    # has waited, smaller share jobs must stop backfilling so room can build up.
    jobs = [FakeJob(1, 1, vram_mb=40_000), FakeJob(2, 1, vram_mb=1024)]
    slots = [GpuSlot(0, 20_000, False)]
    assert plan_ids(s, jobs, slots, now=0.0) == [(2, [0])]
    later = sched.RESERVE_AFTER_S + 1
    assert plan_ids(s, jobs, slots, now=later) == []


# -- free_gpus / capacity --------------------------------------------------


def _gpus(n=3):
    return [GpuInfo(index=i, name="A6000", mem_used_mb=0, mem_total_mb=GPU_MB,
                    util_pct=0, compute_procs={}) for i in range(n)]


def test_free_gpus_empty_machine():
    s = Scheduler()
    slots = s.free_gpus(_gpus(), {}, [])
    assert [(x.index, x.free_mb, x.empty) for x in slots] == [
        (i, GPU_MB - sched.VRAM_HEADROOM_MB, True) for i in range(3)
    ]


def test_free_gpus_subtracts_tenant_budgets():
    s = Scheduler()
    ledger = {0: [Tenant(1, 8192), Tenant(2, 4096)]}
    slot = s.free_gpus(_gpus(), ledger, [])[0]
    expected = GPU_MB - sched.VRAM_HEADROOM_MB - (8192 + 4096) - 2 * sched.VRAM_OVERHEAD_MB
    assert (slot.index, slot.free_mb, slot.empty) == (0, expected, False)


def test_free_gpus_excludes_exclusively_held_gpu():
    s = Scheduler()
    slots = s.free_gpus(_gpus(), {0: [Tenant(1, None)]}, [])
    assert [x.index for x in slots] == [1, 2]


def test_free_gpus_charges_against_budgets_not_live_usage():
    s = Scheduler()
    # Tenant declared 8G but has allocated nothing yet: capacity must still be
    # charged, or the next job would be placed into space already promised.
    gpus = _gpus(1)
    gpus[0].mem_used_mb = 0
    slot = s.free_gpus(gpus, {0: [Tenant(1, 8192)]}, [])[0]
    assert slot.free_mb == GPU_MB - sched.VRAM_HEADROOM_MB - 8192 - sched.VRAM_OVERHEAD_MB


# -- vram policing ---------------------------------------------------------


@dataclass
class RunningJob:
    id: int
    pid: int
    vram_mb: Optional[int] = None


def _gpu_with(procs, index=0):
    return GpuInfo(index=index, name="A6000", mem_used_mb=0, mem_total_mb=GPU_MB,
                   util_pct=0, compute_procs=procs)


def test_no_violation_when_within_budget(monkeypatch):
    monkeypatch.setattr(sched, "_pgid_of", lambda pid: pid)
    job = RunningJob(1, pid=500, vram_mb=8192)
    assert sched.vram_violations([_gpu_with({500: 8000})], [job]) == []


def test_violation_when_over_budget(monkeypatch):
    monkeypatch.setattr(sched, "_pgid_of", lambda pid: pid)
    job = RunningJob(1, pid=500, vram_mb=8192)
    (v,) = sched.vram_violations([_gpu_with({500: 20_000})], [job])
    assert (v.job.id, v.used_mb, v.budget_mb) == (1, 20_000, 8192)


def test_overhead_is_tolerated(monkeypatch):
    monkeypatch.setattr(sched, "_pgid_of", lambda pid: pid)
    job = RunningJob(1, pid=500, vram_mb=8192)
    # Right at budget + overhead is still fine; one MiB past it is not.
    ok = 8192 + sched.VRAM_OVERHEAD_MB
    assert sched.vram_violations([_gpu_with({500: ok})], [job]) == []
    assert sched.vram_violations([_gpu_with({500: ok + 1})], [job])


def test_exclusive_jobs_are_not_policed(monkeypatch):
    monkeypatch.setattr(sched, "_pgid_of", lambda pid: pid)
    job = RunningJob(1, pid=500, vram_mb=None)  # exclusive: owns the whole card
    assert sched.vram_violations([_gpu_with({500: GPU_MB})], [job]) == []


def test_fails_open_when_memory_unmeasurable(monkeypatch):
    monkeypatch.setattr(sched, "_pgid_of", lambda pid: pid)
    job = RunningJob(1, pid=500, vram_mb=1024)
    # NVML reports None for per-process memory under some driver configs; a job
    # we cannot measure must never be killed.
    assert sched.vram_violations([_gpu_with({500: None})], [job]) == []


def test_other_users_processes_are_ignored(monkeypatch):
    monkeypatch.setattr(sched, "_pgid_of", lambda pid: pid)
    job = RunningJob(1, pid=500, vram_mb=8192)
    assert sched.vram_violations([_gpu_with({999: 40_000})], [job]) == []


def test_usage_summed_across_a_jobs_processes(monkeypatch):
    # A job's own children share its process group, so a multi-process job's
    # usage has to be added up before it is compared with the budget.
    monkeypatch.setattr(sched, "_pgid_of", lambda pid: 500 if pid in (500, 501) else pid)
    job = RunningJob(1, pid=500, vram_mb=8192)
    (v,) = sched.vram_violations([_gpu_with({500: 5000, 501: 5000})], [job])
    assert v.used_mb == 10_000


def _gpu(procs):
    return GpuInfo(index=0, name="g", mem_used_mb=0, mem_total_mb=1000,
                   util_pct=0, compute_procs=procs)


def test_externally_busy_ignores_small_contexts():
    assert not externally_busy(_gpu({12345678: 100}), our_pgids=set())


def test_externally_busy_detects_big_external_proc():
    assert externally_busy(_gpu({12345678: 8000}), our_pgids=set())


def test_our_own_jobs_not_external(monkeypatch):
    monkeypatch.setattr(sched, "_pgid_of", lambda pid: 4242)
    assert not externally_busy(_gpu({777: 8000}), our_pgids={4242})
