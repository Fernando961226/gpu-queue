import sqlite3

from gpu_queue import db as dbm
from gpu_queue.db import Database, Tenant

# The jobs table as it shipped before vram_mb existed. Kept verbatim so the
# migration is tested against a real old database, not against _SCHEMA minus a
# line — which would silently start passing if _SCHEMA changed.
_OLD_SCHEMA = """
CREATE TABLE jobs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    name                TEXT NOT NULL,
    command             TEXT NOT NULL,
    workdir             TEXT NOT NULL,
    env                 TEXT NOT NULL,
    conda_env           TEXT,
    gpus_requested      INTEGER NOT NULL,
    gpu_ids             TEXT,
    pid                 INTEGER,
    state               TEXT NOT NULL,
    submitted_at        REAL NOT NULL,
    started_at          REAL,
    finished_at         REAL,
    exit_code           INTEGER,
    log_path            TEXT,
    cancel_requested_at REAL,
    note                TEXT
);
"""


def make_db(tmp_path):
    return Database(tmp_path / "db.sqlite")


def add(db, name="j", gpus=1, vram_mb=None):
    return db.add_job(name, ["echo", "hi"], "/tmp", {"PATH": "/bin"}, gpus,
                      vram_mb=vram_mb)


def test_add_and_get(tmp_path):
    db = make_db(tmp_path)
    job = add(db, "train", gpus=2)
    got = db.get_job(job.id)
    assert got.name == "train"
    assert got.state == dbm.QUEUED
    assert got.command == ["echo", "hi"]
    assert got.gpus_requested == 2
    assert got.env == {"PATH": "/bin"}


def test_lifecycle_and_ledger(tmp_path):
    db = make_db(tmp_path)
    j1, j2 = add(db), add(db)
    db.mark_running(j1.id, pid=1234, gpu_ids=[0, 1], log_path="/tmp/x.log")
    assert db.assigned_gpus() == {0: [Tenant(j1.id)], 1: [Tenant(j1.id)]}
    db.mark_finished(j1.id, dbm.DONE, 0)
    assert db.assigned_gpus() == {}
    got = db.get_job(j1.id)
    assert got.state == dbm.DONE and got.exit_code == 0
    assert db.jobs_in_state(dbm.QUEUED)[0].id == j2.id


def test_migrates_database_from_before_vram_mb(tmp_path):
    """An upgraded daemon must open an old database instead of dying on it."""
    path = tmp_path / "db.sqlite"
    conn = sqlite3.connect(str(path))
    conn.executescript(_OLD_SCHEMA)
    conn.execute(
        "INSERT INTO jobs (name, command, workdir, env, gpus_requested, state,"
        " submitted_at) VALUES ('old', '[\"true\"]', '/tmp', '{}', 1, 'DONE', 1.0)"
    )
    conn.commit()
    conn.close()

    db = Database(path)  # would raise IndexError on the missing column

    job = db.get_job(1)
    assert job.name == "old"
    assert job.vram_mb is None  # NULL reads as an exclusive job
    # and the migrated database is fully usable afterwards
    assert db.get_job(add(db, vram_mb=4096).id).vram_mb == 4096


def test_migration_is_idempotent(tmp_path):
    """Reopening an already-migrated database must not re-run the ALTER."""
    path = tmp_path / "db.sqlite"
    Database(path).close()
    db = Database(path)  # duplicate column error if the guard were missing
    assert db.get_job(add(db, vram_mb=2048).id).vram_mb == 2048


def test_vram_mb_round_trips(tmp_path):
    db = make_db(tmp_path)
    assert db.get_job(add(db, vram_mb=8192).id).vram_mb == 8192
    assert db.get_job(add(db).id).vram_mb is None  # NULL == exclusive job


def test_ledger_tracks_multiple_tenants(tmp_path):
    # The old Dict[gpu, job_id] ledger could not express this: the second job
    # would have overwritten the first.
    db = make_db(tmp_path)
    j1 = add(db, name="share-a", vram_mb=8192)
    j2 = add(db, name="share-b", vram_mb=4096)
    db.mark_running(j1.id, pid=1, gpu_ids=[0], log_path="/tmp/a.log")
    db.mark_running(j2.id, pid=2, gpu_ids=[0], log_path="/tmp/b.log")
    assert db.assigned_gpus() == {
        0: [Tenant(j1.id, 8192), Tenant(j2.id, 4096)],
    }


def test_ledger_drops_tenant_when_job_finishes(tmp_path):
    db = make_db(tmp_path)
    j1 = add(db, vram_mb=8192)
    j2 = add(db, vram_mb=4096)
    db.mark_running(j1.id, pid=1, gpu_ids=[0], log_path="/tmp/a.log")
    db.mark_running(j2.id, pid=2, gpu_ids=[0], log_path="/tmp/b.log")
    db.mark_finished(j1.id, dbm.DONE, 0)
    assert db.assigned_gpus() == {0: [Tenant(j2.id, 4096)]}


def test_requeue_resets(tmp_path):
    db = make_db(tmp_path)
    job = add(db)
    db.mark_running(job.id, 1, [0], "/tmp/x.log")
    db.mark_cancel_requested(job.id)
    db.mark_finished(job.id, dbm.CANCELLED, None)
    db.requeue(job.id)
    got = db.get_job(job.id)
    assert got.state == dbm.QUEUED
    assert got.pid is None and got.gpu_ids is None
    assert got.exit_code is None and got.cancel_requested_at is None


def test_queue_order_by_submit_time(tmp_path):
    db = make_db(tmp_path)
    a, b = add(db, "a"), add(db, "b")
    ids = [j.id for j in db.jobs_in_state(dbm.QUEUED)]
    assert ids == [a.id, b.id]
    # requeueing `a` sends it to the back of the queue
    db.mark_running(a.id, 1, [0], "x")
    db.mark_finished(a.id, dbm.FAILED, 1)
    db.requeue(a.id)
    ids = [j.id for j in db.jobs_in_state(dbm.QUEUED)]
    assert ids == [b.id, a.id]
