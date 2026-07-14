from dataclasses import dataclass

import gpu_queue.scheduler as sched
from gpu_queue.scheduler import GpuInfo, Scheduler, externally_busy


@dataclass
class FakeJob:
    id: int
    gpus_requested: int


def plan_ids(scheduler, jobs, free, now=0.0):
    return [(a.job.id, a.gpu_ids) for a in scheduler.plan(jobs, free, now=now)]


def test_fifo_order():
    s = Scheduler()
    jobs = [FakeJob(1, 1), FakeJob(2, 1), FakeJob(3, 1)]
    assert plan_ids(s, jobs, [0, 1]) == [(1, [0]), (2, [1])]


def test_no_double_assignment():
    s = Scheduler()
    jobs = [FakeJob(1, 2), FakeJob(2, 2)]
    out = plan_ids(s, jobs, [0, 1, 2, 3])
    used = [g for _, gpus in out for g in gpus]
    assert len(used) == len(set(used)) == 4


def test_backfill_when_head_blocked():
    s = Scheduler()
    jobs = [FakeJob(1, 4), FakeJob(2, 1)]
    # head needs 4, only 2 free: job 2 backfills
    assert plan_ids(s, jobs, [0, 1]) == [(2, [0])]


def test_only_first_blocked_job_reserves():
    s = Scheduler()
    jobs = [FakeJob(1, 4), FakeJob(2, 3), FakeJob(3, 1)]
    # head blocked, job 2 also blocked, job 3 backfills
    assert plan_ids(s, jobs, [0, 1]) == [(3, [0])]


def test_reservation_stops_backfill_after_timeout():
    s = Scheduler()
    jobs = [FakeJob(1, 4), FakeJob(2, 1)]
    # first cycle at t=0: head becomes blocked, backfill allowed
    assert plan_ids(s, jobs, [0, 1], now=0.0) == [(2, [0])]
    # much later: reservation kicks in, no more backfill
    later = sched.RESERVE_AFTER_S + 1
    assert plan_ids(s, jobs, [0, 1], now=later) == []


def test_head_runs_when_enough_gpus_free():
    s = Scheduler()
    jobs = [FakeJob(1, 4), FakeJob(2, 1)]
    assert plan_ids(s, jobs, [0, 1], now=0.0) == [(2, [0])]
    # head finally fits: it runs and its blocked-tracking is dropped
    out = plan_ids(s, jobs, [0, 1, 2, 3, 4], now=sched.RESERVE_AFTER_S + 1)
    assert out == [(1, [0, 1, 2, 3]), (2, [4])]
    assert 1 not in s._head_blocked_since


def test_blocked_tracking_cleared_when_job_leaves_queue():
    s = Scheduler()
    plan_ids(s, [FakeJob(1, 4)], [0], now=0.0)
    assert 1 in s._head_blocked_since
    plan_ids(s, [], [0], now=1.0)  # job cancelled
    assert 1 not in s._head_blocked_since


def test_zero_gpu_job_runs_even_with_nothing_free():
    s = Scheduler()
    assert plan_ids(s, [FakeJob(1, 0)], []) == [(1, [])]


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
