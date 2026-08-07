"""The gpu-queue daemon (`gqd`): scheduler loop, dispatch, reaping, recovery.

Jobs are children of the daemon but live in their own sessions, so they
survive daemon restarts. On startup we reconcile the DB against live pids:
still-running jobs are re-adopted (tracked by pid + exit file instead of a
Popen handle), dead ones are finalized from their persisted exit code.
"""

import logging
import os
import signal
import subprocess
import sys
import threading
import time
from typing import Dict, Optional

from . import __version__, api_port, db_path, gq_home
from . import db as dbm
from .api import serve
from .db import Database, Job
from .runner import group_alive, job_log_path, kill_tree, launch, read_exit_code
from .scheduler import (
    VRAM_HEADROOM_MB,
    VRAM_OVERHEAD_MB,
    Scheduler,
    _our_pgids,
    externally_busy,
    make_monitor,
    vram_violations,
)

POLL_S = float(os.environ.get("GQ_POLL_S", "2"))
CANCEL_GRACE_S = float(os.environ.get("GQ_CANCEL_GRACE_S", "10"))
# Consecutive cycles a job must be over its declared VRAM before it is evicted.
# One reading can catch a transient spike between a free and the allocator
# returning it; a job that is genuinely over stays over.
VRAM_STRIKES = int(os.environ.get("GQ_VRAM_STRIKES", "3"))

log = logging.getLogger("gqd")


class Daemon:
    def __init__(self, db: Database):
        self.db = db
        self.monitor = make_monitor()
        self.scheduler = Scheduler()
        self.popens: Dict[int, subprocess.Popen] = {}
        self.lock = threading.RLock()  # guards job state transitions
        self._stop = threading.Event()
        self._stragglers_killed: set = set()
        # job id -> consecutive cycles seen over its declared VRAM
        self._vram_strikes: Dict[int, int] = {}
        # job id -> why it was evicted, so _finalize can mark it FAILED rather
        # than CANCELLED even though eviction goes through the cancel path
        self._evicted: Dict[int, str] = {}
        self.last_gpu_snapshot = self.monitor.snapshot()

    # -- actions (called from API threads and the loop) ---------------------

    def submit(self, name, command, workdir, env, gpus, conda_env=None, vram_mb=None) -> Job:

        if vram_mb is not None:
            if gpus>1:
                raise ValueError("--vram jobs use exactly one GPU")
            biggest = max((g.mem_total_mb for g in self.last_gpu_snapshot), default=0)
            if vram_mb > biggest:
                    raise ValueError(f"job wants {vram_mb} MiB but the largest GPU has {biggest} MiB")

        if gpus < 0:
            raise ValueError("gpus must be >= 0")
        total = len(self.last_gpu_snapshot)
        if gpus > total:
            raise ValueError(
                f"job wants {gpus} GPUs but this machine only has {total}"
            )
        if not command:
            raise ValueError("empty command")
        if not os.path.isdir(workdir):
            raise ValueError(f"workdir does not exist: {workdir}")
        with self.lock:
            job = self.db.add_job(name, command, workdir, env, gpus, conda_env, vram_mb)
        log.info("job %d submitted: %s (gpus=%d)", job.id, name, gpus)
        return job

    def cancel(self, job_id: int) -> Job:
        with self.lock:
            job = self.db.get_job(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.state == dbm.QUEUED:
                self.db.mark_finished(job_id, dbm.CANCELLED, None)
            elif job.state == dbm.RUNNING:
                # SIGTERM the whole tree; the reaper finalizes to CANCELLED
                # when it dies, and escalates to SIGKILL after the grace period.
                self.db.mark_cancel_requested(job_id)
                if job.pid:
                    kill_tree(job.pid, signal.SIGTERM)
                log.info("job %d cancel requested (pid %s)", job_id, job.pid)
            # cancel on a finished job is a no-op
            return self.db.get_job(job_id)

    def requeue(self, job_id: int) -> Job:
        with self.lock:
            job = self.db.get_job(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.state not in dbm.FINAL_STATES:
                raise ValueError(f"job {job_id} is {job.state}; only finished jobs can be requeued")
            self.db.requeue(job_id)
            log.info("job %d requeued", job_id)
            return self.db.get_job(job_id)

    def gpu_status(self) -> list:
        """Per-GPU view for `gq gpus` and the extension."""
        with self.lock:
            running = self.db.jobs_in_state(dbm.RUNNING)
            ledger = self.db.assigned_gpus()
        ours = _our_pgids(running)
        names = {j.id: j.name for j in running}
        out = []

        for g in self.last_gpu_snapshot:
            tenants = ledger.get(g.index, [])
            if tenants:
                state = "allocated"
            elif externally_busy(g, ours):
                state = "external"
            else:
                state = "free"
            reserved = sum(
                t.vram_mb + VRAM_OVERHEAD_MB for t in tenants if t.vram_mb is not None
            )
            exclusive = any(t.vram_mb is None for t in tenants)
            out.append(
                {
                    "index": g.index,
                    "name": g.name,
                    "mem_used_mb": g.mem_used_mb,
                    "mem_total_mb": g.mem_total_mb,
                    "util_pct": g.util_pct,
                    "state": state,
                    # Kept for older clients: the first tenant, or None.
                    "job_id": tenants[0].job_id if tenants else None,
                    "tenants": [
                        {
                            "job_id": t.job_id,
                            "name": names.get(t.job_id, ""),
                            "vram_mb": t.vram_mb,
                        }
                        for t in tenants
                    ],
                    # Capacity for further share jobs. An exclusively-held card
                    # has none, however little it is actually using.
                    "vram_capacity_mb": max(0, g.mem_total_mb - VRAM_HEADROOM_MB),
                    "vram_reserved_mb": (
                        max(0, g.mem_total_mb - VRAM_HEADROOM_MB) if exclusive else reserved
                    ),
                }
            )
        return out

    def request_shutdown(self) -> None:
        self._stop.set()

    # -- recovery ------------------------------------------------------------

    def recover(self) -> None:
        """Reconcile DB vs live pids after a daemon restart."""
        for job in self.db.jobs_in_state(dbm.RUNNING):
            exit_code = read_exit_code(job.id)
            # The exit file is authoritative: if it exists the job finished,
            # even if the pid "looks" alive (it may have been reused).
            if exit_code is None and job.pid and group_alive(job.pid):
                log.info("job %d re-adopted (pid %d still running)", job.id, job.pid)
                continue  # reaper tracks it via pid + exit file
            self._finalize(job, exit_code, note="finished while daemon was down")

    # -- loop ----------------------------------------------------------------

    def _finalize(self, job: Job, exit_code: Optional[int], note: Optional[str] = None) -> None:
        evicted = self._evicted.pop(job.id, None)
        if evicted is not None:
            # Eviction reuses the cancel path to get SIGTERM->SIGKILL, but the
            # user did not cancel this: it is a failure, and the note has to say
            # why or `gq ls` just shows a job that mysteriously died.
            state, note = dbm.FAILED, evicted
        elif job.cancel_requested_at is not None:
            state = dbm.CANCELLED
        elif exit_code == 0:
            state = dbm.DONE
        else:
            state = dbm.FAILED
            if exit_code is None:
                note = note or "exit code unknown (process killed?)"
        self.db.mark_finished(job.id, state, exit_code, note)
        self.popens.pop(job.id, None)
        self._stragglers_killed.discard(job.id)
        self._vram_strikes.pop(job.id, None)
        log.info("job %d finished: %s (exit=%s)", job.id, state, exit_code)

    def _enforce_vram(self) -> None:
        """Evict share jobs that exceed the VRAM budget they declared."""
        running = self.db.jobs_in_state(dbm.RUNNING)
        over = {v.job.id: v for v in vram_violations(self.last_gpu_snapshot, running)}

        # Reset anyone who came back under budget: strikes must be consecutive.
        for jid in list(self._vram_strikes):
            if jid not in over:
                del self._vram_strikes[jid]

        for jid, v in over.items():
            if jid in self._evicted:
                continue  # already being killed
            strikes = self._vram_strikes.get(jid, 0) + 1
            self._vram_strikes[jid] = strikes
            log.warning(
                "job %d over declared VRAM: %d MiB used vs %d MiB declared (strike %d/%d)",
                jid, v.used_mb, v.budget_mb, strikes, VRAM_STRIKES,
            )
            if strikes < VRAM_STRIKES:
                continue
            reason = (
                f"evicted: used {v.used_mb / 1024:.1f} GiB of VRAM but declared "
                f"{v.budget_mb / 1024:.1f} GiB. Resubmit with a larger --vram."
            )
            log.error("job %d %s", jid, reason)
            self._evicted[jid] = reason
            del self._vram_strikes[jid]
            # Reuse the cancel path so SIGTERM->SIGKILL escalation comes free;
            # _finalize turns it into FAILED rather than CANCELLED.
            self.db.mark_cancel_requested(jid)
            if v.job.pid:
                kill_tree(v.job.pid, signal.SIGTERM)

    def _reap(self) -> None:
        now = time.time()
        for job in self.db.jobs_in_state(dbm.RUNNING):
            popen = self.popens.get(job.id)
            rc = popen.poll() if popen is not None else None  # reaps the zombie
            file_code = read_exit_code(job.id)
            wrapper_done = file_code is not None or rc is not None

            if popen is None and wrapper_done:
                # Adopted job whose wrapper finished. Don't trust the pid —
                # it may have been reused by an unrelated process — so
                # finalize from the exit file and touch nothing.
                self._finalize(job, file_code)
            elif job.pid is None or not group_alive(job.pid):
                # Whole process tree is gone: the job is truly over.
                if file_code is None:
                    file_code = read_exit_code(job.id)  # may have just landed
                if file_code is None and popen is not None:
                    file_code = popen.poll()
                self._finalize(job, file_code)
            elif (
                wrapper_done
                and job.cancel_requested_at is None
                and job.id not in self._stragglers_killed
            ):
                # Script finished but left children behind (e.g. `cmd &`
                # without wait): slurm-style cleanup of the rest of the tree.
                log.info("job %d script exited, killing leftover processes", job.id)
                self._stragglers_killed.add(job.id)
                kill_tree(job.pid, signal.SIGKILL)
            elif (
                job.cancel_requested_at is not None
                and now - job.cancel_requested_at > CANCEL_GRACE_S
            ):
                log.info("job %d ignored SIGTERM for %.0fs, sending SIGKILL",
                         job.id, CANCEL_GRACE_S)
                kill_tree(job.pid, signal.SIGKILL)

    def _dispatch(self) -> None:
        queued = self.db.jobs_in_state(dbm.QUEUED)
        running = self.db.jobs_in_state(dbm.RUNNING)
        ledger = self.db.assigned_gpus()
        free = self.scheduler.free_gpus(self.last_gpu_snapshot, ledger, running)
        for assignment in self.scheduler.plan(queued, free):
            job, gpu_ids = assignment.job, assignment.gpu_ids
            try:
                proc = launch(job, gpu_ids)
            except OSError as e:
                self.db.mark_finished(job.id, dbm.FAILED, None, note=f"launch failed: {e}")
                log.error("job %d launch failed: %s", job.id, e)
                continue
            self.popens[job.id] = proc
            self.db.mark_running(job.id, proc.pid, gpu_ids, str(job_log_path(job.id)))
            log.info("job %d started on gpus %s (pid %d)", job.id, gpu_ids, proc.pid)

    def cycle(self) -> None:
        with self.lock:
            try:
                self.last_gpu_snapshot = self.monitor.snapshot()
            except Exception as e:  # NVML hiccup: keep last snapshot, skip dispatch
                log.warning("GPU snapshot failed: %s", e)
                self._reap()
                return
            self._reap()
            self._enforce_vram()
            self._dispatch()

    def run(self) -> None:
        log.info("gqd %s starting (home=%s, port=%d)", __version__, gq_home(), api_port())
        self.recover()
        server = serve(self)
        log.info("API listening on 127.0.0.1:%d", api_port())
        try:
            while not self._stop.is_set():
                self.cycle()
                self._stop.wait(POLL_S)
        finally:
            server.shutdown()
            log.info("gqd stopped (running jobs keep running)")


def main(argv=None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        stream=sys.stderr,
    )
    gq_home().mkdir(parents=True, exist_ok=True)
    daemon = Daemon(Database(db_path()))

    def _sig(signum, frame):
        daemon.request_shutdown()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    daemon.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
