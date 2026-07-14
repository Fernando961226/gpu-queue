"""Process spawn and kill for jobs.

Each job runs in its own session (setsid) so the whole process tree can be
killed via its process group. The command is wrapped in a tiny bash stub
that records the exit code to <log_dir>/<id>.exit — that file is how a
restarted daemon recovers the outcome of jobs whose Popen handle it lost.
"""

import os
import signal
import subprocess
import time
from pathlib import Path
from typing import List, Optional

from . import log_dir
from .db import Job

# `"$@"` runs the job argv verbatim; the exit code is persisted before exiting
# so a restarted daemon (which can't waitpid a non-child) can still see it.
_WRAPPER = '"$@"; ec=$?; echo "$ec" > "$GQ_EXIT_FILE"; exit "$ec"'


def job_log_path(job_id: int) -> Path:
    return log_dir() / f"{job_id}.log"


def job_exit_path(job_id: int) -> Path:
    return log_dir() / f"{job_id}.exit"


def launch(job: Job, gpu_ids: List[int]) -> subprocess.Popen:
    """Spawn a job on the given GPUs. Raises OSError on spawn failure."""
    log_dir().mkdir(parents=True, exist_ok=True)
    exit_file = job_exit_path(job.id)
    exit_file.unlink(missing_ok=True)

    env = dict(job.env)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
    env["GQ_JOB_ID"] = str(job.id)
    env["GQ_EXIT_FILE"] = str(exit_file)

    log_path = job_log_path(job.id)
    log_fh = open(log_path, "ab", buffering=0)
    header = (
        f"=== gq job {job.id} ({job.name}) ===\n"
        f"=== started {time.strftime('%Y-%m-%d %H:%M:%S')}"
        f"  gpus={gpu_ids}  cwd={job.workdir} ===\n"
        f"=== command: {' '.join(job.command)} ===\n"
    )
    log_fh.write(header.encode())

    try:
        proc = subprocess.Popen(
            ["/bin/bash", "-c", _WRAPPER, "gq-job"] + list(job.command),
            cwd=job.workdir,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        log_fh.close()  # the child holds its own fd now
    return proc


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def kill_tree(pid: int, sig: int = signal.SIGTERM) -> None:
    """Signal the job's whole process group (pgid == pid, we setsid'd)."""
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def read_exit_code(job_id: int) -> Optional[int]:
    """Exit code persisted by the wrapper, or None if it never got written
    (e.g. the process group was SIGKILLed)."""
    try:
        text = job_exit_path(job_id).read_text().strip()
        return int(text)
    except (FileNotFoundError, ValueError):
        return None
