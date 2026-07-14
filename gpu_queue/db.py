"""SQLite job store.

Only the daemon process opens the database; CLI and extension go through the
HTTP API. A single connection guarded by a lock is shared between the
scheduler thread and the API threads.
"""

import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# Job lifecycle: QUEUED -> RUNNING -> DONE | FAILED | CANCELLED
QUEUED = "QUEUED"
RUNNING = "RUNNING"
DONE = "DONE"
FAILED = "FAILED"
CANCELLED = "CANCELLED"

ACTIVE_STATES = (QUEUED, RUNNING)
FINAL_STATES = (DONE, FAILED, CANCELLED)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    command             TEXT NOT NULL,          -- JSON list of argv words
    workdir             TEXT NOT NULL,
    env                 TEXT NOT NULL,          -- JSON dict, environment snapshot
    conda_env           TEXT,                   -- display only (CONDA_DEFAULT_ENV)
    gpus_requested      INTEGER NOT NULL,
    gpu_ids             TEXT,                   -- JSON list of assigned GPU indices
    pid                 INTEGER,                -- process-group leader pid
    state               TEXT NOT NULL,
    submitted_at        REAL NOT NULL,
    started_at          REAL,
    finished_at         REAL,
    exit_code           INTEGER,
    log_path            TEXT,
    cancel_requested_at REAL,
    note                TEXT                    -- human-readable detail (failures etc.)
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state);
"""


@dataclass
class Job:
    id: int
    name: str
    command: List[str]
    workdir: str
    env: Dict[str, str]
    conda_env: Optional[str]
    gpus_requested: int
    gpu_ids: Optional[List[int]]
    pid: Optional[int]
    state: str
    submitted_at: float
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    exit_code: Optional[int] = None
    log_path: Optional[str] = None
    cancel_requested_at: Optional[float] = None
    note: Optional[str] = None

    def to_dict(self) -> dict:
        d = self.__dict__.copy()
        return d


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        name=row["name"],
        command=json.loads(row["command"]),
        workdir=row["workdir"],
        env=json.loads(row["env"]),
        conda_env=row["conda_env"],
        gpus_requested=row["gpus_requested"],
        gpu_ids=json.loads(row["gpu_ids"]) if row["gpu_ids"] else None,
        pid=row["pid"],
        state=row["state"],
        submitted_at=row["submitted_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        exit_code=row["exit_code"],
        log_path=row["log_path"],
        cancel_requested_at=row["cancel_requested_at"],
        note=row["note"],
    )


class Database:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()
        self._lock = threading.RLock()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- writes ------------------------------------------------------------

    def add_job(
        self,
        name: str,
        command: List[str],
        workdir: str,
        env: Dict[str, str],
        gpus_requested: int,
        conda_env: Optional[str] = None,
    ) -> Job:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO jobs (name, command, workdir, env, conda_env,"
                " gpus_requested, state, submitted_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    name,
                    json.dumps(command),
                    workdir,
                    json.dumps(env),
                    conda_env,
                    gpus_requested,
                    QUEUED,
                    time.time(),
                ),
            )
            self._conn.commit()
            return self.get_job(cur.lastrowid)

    def mark_running(self, job_id: int, pid: int, gpu_ids: List[int], log_path: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET state=?, pid=?, gpu_ids=?, log_path=?,"
                " started_at=?, note=NULL WHERE id=?",
                (RUNNING, pid, json.dumps(gpu_ids), log_path, time.time(), job_id),
            )
            self._conn.commit()

    def mark_finished(self, job_id: int, state: str, exit_code: Optional[int],
                      note: Optional[str] = None) -> None:
        assert state in FINAL_STATES
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET state=?, exit_code=?, finished_at=?, note=?"
                " WHERE id=?",
                (state, exit_code, time.time(), note, job_id),
            )
            self._conn.commit()

    def mark_cancel_requested(self, job_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET cancel_requested_at=? WHERE id=?",
                (time.time(), job_id),
            )
            self._conn.commit()

    def requeue(self, job_id: int) -> None:
        """Reset a finished job back to QUEUED (same id, log is appended to)."""
        with self._lock:
            self._conn.execute(
                "UPDATE jobs SET state=?, pid=NULL, gpu_ids=NULL, started_at=NULL,"
                " finished_at=NULL, exit_code=NULL, cancel_requested_at=NULL,"
                " note=NULL, submitted_at=? WHERE id=?",
                (QUEUED, time.time(), job_id),
            )
            self._conn.commit()

    # -- reads -------------------------------------------------------------

    def get_job(self, job_id: int) -> Optional[Job]:
        with self._lock:
            row = self._conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return _row_to_job(row) if row else None

    def jobs_in_state(self, *states: str) -> List[Job]:
        """Jobs in the given states, queued-order (submit time, then id)."""
        q = ",".join("?" for _ in states)
        with self._lock:
            rows = self._conn.execute(
                f"SELECT * FROM jobs WHERE state IN ({q})"
                " ORDER BY submitted_at ASC, id ASC",
                states,
            ).fetchall()
        return [_row_to_job(r) for r in rows]

    def list_jobs(self, states: Optional[List[str]] = None, limit: Optional[int] = None) -> List[Job]:
        """Newest-first listing for the API/CLI."""
        sql = "SELECT * FROM jobs"
        args: list = []
        if states:
            sql += " WHERE state IN (%s)" % ",".join("?" for _ in states)
            args += states
        sql += " ORDER BY id DESC"
        if limit:
            sql += " LIMIT ?"
            args.append(limit)
        with self._lock:
            rows = self._conn.execute(sql, args).fetchall()
        return [_row_to_job(r) for r in rows]

    def assigned_gpus(self) -> Dict[int, int]:
        """Map of gpu index -> job id for RUNNING jobs (the allocation ledger)."""
        ledger: Dict[int, int] = {}
        for job in self.jobs_in_state(RUNNING):
            for g in job.gpu_ids or []:
                ledger[g] = job.id
        return ledger
