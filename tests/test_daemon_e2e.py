"""End-to-end tests: real daemon loop + HTTP API + real subprocesses,
with fake GPUs (GQ_FAKE_GPUS) so they run on any machine."""

import json
import os
import socket
import threading
import time
import urllib.error
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


class ApiFailure(AssertionError):
    """A non-2xx response, carrying the daemon's own error text.

    Subclasses AssertionError so pytest reports it as a plain test failure,
    and keeps `.code` so tests can assert on the status.
    """

    def __init__(self, method, path, code, body):
        self.code = code
        self.body = body
        super().__init__(f"{method} {path} -> {code}: {body}")


def api(method, path, body=None):
    from gpu_queue import api_url

    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(api_url() + path, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # The daemon puts the real reason in the body (api.py catches and
        # serializes it); HTTPError carries only the status line, so without
        # this every failure reads as a bare "500 Internal Server Error".
        raise ApiFailure(method, path, e.code, e.read().decode()) from None


def submit(command, gpus=1, name="test", workdir=None, vram_mb=None):
    return api("POST", "/api/submit", {
        "name": name,
        "command": command,
        "workdir": workdir or os.getcwd(),
        "env": dict(os.environ),
        "gpus": gpus,
        "vram_mb": vram_mb,
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
    with pytest.raises(ApiFailure) as exc:
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


# -- vram sharing ----------------------------------------------------------


def test_share_jobs_pack_onto_one_gpu(gq):
    # FakeGPU is 24000MB, so two 4G jobs comfortably share one card while the
    # other stays empty.
    a = submit(["sleep", "30"], name="a", vram_mb=4096)
    b = submit(["sleep", "30"], name="b", vram_mb=4096)
    wait_for(lambda: all(job_state(j["id"]) == dbm.RUNNING for j in (a, b)),
             msg="both RUNNING")
    gpus_a = api("GET", f"/api/jobs/{a['id']}")["gpu_ids"]
    gpus_b = api("GET", f"/api/jobs/{b['id']}")["gpu_ids"]
    assert gpus_a == gpus_b, "share jobs should land on the same GPU (best-fit)"


def test_exclusive_job_avoids_shared_gpu(gq):
    share = submit(["sleep", "30"], name="share", vram_mb=4096)
    wait_for(lambda: job_state(share["id"]) == dbm.RUNNING, msg="share RUNNING")
    shared_gpu = api("GET", f"/api/jobs/{share['id']}")["gpu_ids"][0]

    excl = submit(["sleep", "30"], name="excl", gpus=1)
    wait_for(lambda: job_state(excl["id"]) == dbm.RUNNING, msg="exclusive RUNNING")
    assert api("GET", f"/api/jobs/{excl['id']}")["gpu_ids"] != [shared_gpu]


def test_share_job_waits_when_no_capacity(gq):
    # 2 fake GPUs of 24000MB; headroom leaves ~23000 each. A 20G job on each
    # fills both, so a third has nowhere to go.
    for i in range(2):
        submit(["sleep", "30"], name=f"big{i}", vram_mb=20 * 1024)
    wait_for(lambda: len(gq.db.jobs_in_state(dbm.RUNNING)) == 2, msg="two RUNNING")
    late = submit(["sleep", "30"], name="late", vram_mb=20 * 1024)
    time.sleep(0.5)
    assert job_state(late["id"]) == dbm.QUEUED


def test_vram_over_budget_job_is_evicted(gq, monkeypatch):
    monkeypatch.setattr(daemon_mod, "VRAM_STRIKES", 2)
    job = submit(["sleep", "30"], name="hog", vram_mb=1024)
    wait_for(lambda: job_state(job["id"]) == dbm.RUNNING, msg="RUNNING")

    # Pretend NVML sees this job's process group using far more than declared.
    got = api("GET", f"/api/jobs/{job['id']}")
    gq.monitor.fake_procs[got["gpu_ids"][0]] = {got["pid"]: 20_000}

    wait_for(lambda: job_state(job["id"]) == dbm.FAILED, msg="evicted", timeout=15)
    got = api("GET", f"/api/jobs/{job['id']}")
    assert "evicted" in (got["note"] or "")
    assert got["state"] == dbm.FAILED  # not CANCELLED: the user did not cancel


def test_vram_within_budget_is_not_evicted(gq, monkeypatch):
    monkeypatch.setattr(daemon_mod, "VRAM_STRIKES", 2)
    job = submit(["sleep", "3"], name="honest", vram_mb=8192)
    wait_for(lambda: job_state(job["id"]) == dbm.RUNNING, msg="RUNNING")
    got = api("GET", f"/api/jobs/{job['id']}")
    gq.monitor.fake_procs[got["gpu_ids"][0]] = {got["pid"]: 8000}  # under budget
    wait_for(lambda: job_state(job["id"]) == dbm.DONE, msg="DONE", timeout=15)


def test_unmeasurable_vram_never_evicts(gq, monkeypatch):
    monkeypatch.setattr(daemon_mod, "VRAM_STRIKES", 2)
    job = submit(["sleep", "3"], name="opaque", vram_mb=1024)
    wait_for(lambda: job_state(job["id"]) == dbm.RUNNING, msg="RUNNING")
    got = api("GET", f"/api/jobs/{job['id']}")
    gq.monitor.fake_procs[got["gpu_ids"][0]] = {got["pid"]: None}  # NVML cannot read it: must fail open
    wait_for(lambda: job_state(job["id"]) == dbm.DONE, msg="DONE", timeout=15)


def test_exclusive_job_is_not_policed(gq, monkeypatch):
    monkeypatch.setattr(daemon_mod, "VRAM_STRIKES", 2)
    job = submit(["sleep", "3"], name="excl", gpus=1)
    wait_for(lambda: job_state(job["id"]) == dbm.RUNNING, msg="RUNNING")
    got = api("GET", f"/api/jobs/{job['id']}")
    gq.monitor.fake_procs[got["gpu_ids"][0]] = {got["pid"]: 23_000}  # owns the card, so this is fine
    wait_for(lambda: job_state(job["id"]) == dbm.DONE, msg="DONE", timeout=15)


def test_submit_vram_with_multiple_gpus_rejected(gq):
    with pytest.raises(ApiFailure) as exc:
        submit(["true"], gpus=2, vram_mb=4096)
    assert exc.value.code == 400
    assert "one GPU" in exc.value.body


def test_submit_vram_larger_than_any_gpu_rejected(gq):
    with pytest.raises(ApiFailure) as exc:
        submit(["true"], vram_mb=999_999)
    assert exc.value.code == 400


def test_gpus_endpoint_reports_tenants(gq):
    job = submit(["sleep", "30"], name="tenant", vram_mb=4096)
    wait_for(lambda: job_state(job["id"]) == dbm.RUNNING, msg="RUNNING")
    gpus = api("GET", "/api/gpus")
    holding = [g for g in gpus if g["tenants"]]
    assert len(holding) == 1
    (t,) = holding[0]["tenants"]
    assert (t["job_id"], t["name"], t["vram_mb"]) == (job["id"], "tenant", 4096)
    assert holding[0]["vram_reserved_mb"] > 4096  # budget + overhead
