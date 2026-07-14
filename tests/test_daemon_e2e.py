"""End-to-end tests: real daemon loop + HTTP API + real subprocesses,
with fake GPUs (GQ_FAKE_GPUS) so they run on any machine."""

import json
import os
import socket
import threading
import time
import urllib.request

import pytest

import gpu_queue.daemon as daemon_mod
from gpu_queue import db as dbm
from gpu_queue.daemon import Daemon
from gpu_queue.db import Database
from gpu_queue.runner import group_alive, pid_alive


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def gq(tmp_path, monkeypatch):
    monkeypatch.setenv("GQ_HOME", str(tmp_path))
    monkeypatch.setenv("GQ_FAKE_GPUS", "2")
    monkeypatch.setenv("GQ_PORT", str(free_port()))
    monkeypatch.setattr(daemon_mod, "POLL_S", 0.05)
    monkeypatch.setattr(daemon_mod, "CANCEL_GRACE_S", 1.0)
    d = Daemon(Database(tmp_path / "db.sqlite"))
    t = threading.Thread(target=d.run, daemon=True)
    t.start()
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            api("GET", "/api/health")
            break
        except Exception:
            time.sleep(0.05)
    yield d
    d.request_shutdown()
    t.join(timeout=5)


def api(method, path, body=None):
    from gpu_queue import api_url

    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(api_url() + path, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def submit(command, gpus=1, name="test", workdir=None):
    return api("POST", "/api/submit", {
        "name": name,
        "command": command,
        "workdir": workdir or os.getcwd(),
        "env": dict(os.environ),
        "gpus": gpus,
    })


def wait_for(cond, timeout=10, msg="condition"):
    deadline = time.time() + timeout
    while time.time() < deadline:
        v = cond()
        if v:
            return v
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {msg}")


def job_state(job_id):
    return api("GET", f"/api/jobs/{job_id}")["state"]


def test_job_runs_to_done_with_cuda_devices(gq, tmp_path):
    job = submit(["bash", "-c", "echo dev=$CUDA_VISIBLE_DEVICES"], gpus=2)
    wait_for(lambda: job_state(job["id"]) == dbm.DONE, msg="job DONE")
    done = api("GET", f"/api/jobs/{job['id']}")
    assert done["exit_code"] == 0
    log = (tmp_path / "logs" / f"{job['id']}.log").read_text()
    assert "dev=0,1" in log


def test_failed_job(gq):
    job = submit(["bash", "-c", "exit 3"])
    wait_for(lambda: job_state(job["id"]) == dbm.FAILED, msg="job FAILED")
    assert api("GET", f"/api/jobs/{job['id']}")["exit_code"] == 3


def test_fifo_queueing_two_gpus(gq):
    a = submit(["sleep", "30"], name="a")
    b = submit(["sleep", "30"], name="b")
    c = submit(["sleep", "30"], name="c")
    wait_for(lambda: job_state(a["id"]) == dbm.RUNNING and job_state(b["id"]) == dbm.RUNNING,
             msg="a and b RUNNING")
    time.sleep(0.3)
    assert job_state(c["id"]) == dbm.QUEUED  # both GPUs taken
    ledger = {g["index"]: g for g in api("GET", "/api/gpus")}
    assert {ledger[0]["state"], ledger[1]["state"]} == {"allocated"}
    for jid in (a["id"], b["id"], c["id"]):
        api("POST", f"/api/jobs/{jid}/cancel")


def test_cancel_escalates_to_sigkill_for_term_ignoring_jobs(gq):
    # a child that ignores SIGTERM must keep the job RUNNING through the
    # grace period, then be SIGKILLed and only then marked CANCELLED
    job = submit(["bash", "-c", 'trap "" TERM; while :; do sleep 1; done'])
    wait_for(lambda: job_state(job["id"]) == dbm.RUNNING, msg="RUNNING")
    pid = api("GET", f"/api/jobs/{job['id']}")["pid"]
    api("POST", f"/api/jobs/{job['id']}/cancel")
    time.sleep(0.3)
    assert job_state(job["id"]) == dbm.RUNNING  # inside grace, TERM ignored
    wait_for(lambda: job_state(job["id"]) == dbm.CANCELLED, msg="CANCELLED after grace")
    wait_for(lambda: not group_alive(pid), timeout=5, msg="process group dead")


def test_straggler_children_killed_when_script_exits(gq):
    # script backgrounds a child and exits; the child must be cleaned up
    # before the job is finalized (slurm-style)
    job = submit(["bash", "-c", "sleep 60 & echo bye"])
    wait_for(lambda: job_state(job["id"]) == dbm.DONE, msg="DONE")
    pid = api("GET", f"/api/jobs/{job['id']}")["pid"]
    assert not group_alive(pid)
    assert api("GET", f"/api/jobs/{job['id']}")["exit_code"] == 0


def test_submit_more_gpus_than_machine_rejected(gq):
    import urllib.error

    with pytest.raises(urllib.error.HTTPError) as exc:
        submit(["true"], gpus=99)
    assert exc.value.code == 400


def test_cancel_kills_process_tree(gq):
    # parent bash spawns a child sleep; cancelling must kill both
    job = submit(["bash", "-c", "sleep 60 & wait"])
    wait_for(lambda: job_state(job["id"]) == dbm.RUNNING, msg="RUNNING")
    pid = api("GET", f"/api/jobs/{job['id']}")["pid"]
    api("POST", f"/api/jobs/{job['id']}/cancel")
    wait_for(lambda: job_state(job["id"]) == dbm.CANCELLED, msg="CANCELLED")
    wait_for(lambda: not pid_alive(pid), timeout=5, msg="process dead")


def test_requeue_runs_again(gq):
    job = submit(["bash", "-c", "echo run-$GQ_JOB_ID"])
    wait_for(lambda: job_state(job["id"]) == dbm.DONE, msg="DONE")
    api("POST", f"/api/jobs/{job['id']}/requeue")
    wait_for(lambda: job_state(job["id"]) == dbm.DONE, msg="DONE again")


def test_externally_busy_gpus_not_used(gq, monkeypatch):
    monkeypatch.setenv("GQ_FAKE_BUSY", "0,1")
    time.sleep(0.2)  # let a snapshot with busy GPUs land
    job = submit(["true"])
    time.sleep(0.5)
    assert job_state(job["id"]) == dbm.QUEUED
    gpus = api("GET", "/api/gpus")
    assert all(g["state"] == "external" for g in gpus)
    monkeypatch.delenv("GQ_FAKE_BUSY")
    wait_for(lambda: job_state(job["id"]) == dbm.DONE, msg="DONE after GPUs freed")


def test_recovery_finalizes_dead_job(gq, tmp_path):
    """Simulate a daemon restart: a RUNNING job whose pid is gone gets
    finalized from its exit file."""
    db = gq.db
    job = db.add_job("ghost", ["true"], "/tmp", {}, 1)
    (tmp_path / "logs").mkdir(exist_ok=True)
    (tmp_path / "logs" / f"{job.id}.exit").write_text("0\n")
    db.mark_running(job.id, pid=999999999, gpu_ids=[0], log_path="x")
    gq.recover()
    assert db.get_job(job.id).state == dbm.DONE


def test_logs_endpoint_tail(gq):
    job = submit(["bash", "-c", "echo hello-tail"])
    wait_for(lambda: job_state(job["id"]) == dbm.DONE, msg="DONE")
    from gpu_queue import api_url

    with urllib.request.urlopen(api_url() + f"/api/jobs/{job['id']}/logs") as resp:
        text = resp.read().decode()
        assert "hello-tail" in text
        assert int(resp.headers["X-Log-Offset"]) == len(text.encode())
