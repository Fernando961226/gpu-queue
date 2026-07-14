from gpu_queue import db as dbm
from gpu_queue.db import Database


def make_db(tmp_path):
    return Database(tmp_path / "db.sqlite")


def add(db, name="j", gpus=1):
    return db.add_job(name, ["echo", "hi"], "/tmp", {"PATH": "/bin"}, gpus)


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
    assert db.assigned_gpus() == {0: j1.id, 1: j1.id}
    db.mark_finished(j1.id, dbm.DONE, 0)
    assert db.assigned_gpus() == {}
    got = db.get_job(j1.id)
    assert got.state == dbm.DONE and got.exit_code == 0
    assert db.jobs_in_state(dbm.QUEUED)[0].id == j2.id


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
