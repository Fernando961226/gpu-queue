"""`gq` — the gpu-queue CLI. Talks to the daemon over the localhost API."""

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import __version__, api_url, gq_home
from .jobscript import JobScriptError, parse_file

# environment variables that describe the submitting shell, not the job
_ENV_SKIP = {"CUDA_VISIBLE_DEVICES", "GQ_JOB_ID", "GQ_EXIT_FILE"}

SYSTEMD_UNIT = "gpu-queue.service"


class CliError(Exception):
    pass


# -- HTTP client ---------------------------------------------------------


def _request(method: str, path: str, body: dict = None, raw: bool = False):
    url = api_url() + path
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data:
        req.add_header("Content-Type", "application/json")
    try:
        resp = urllib.request.urlopen(req, timeout=30)
    except urllib.error.HTTPError as e:
        try:
            msg = json.loads(e.read()).get("error", str(e))
        except Exception:
            msg = str(e)
        raise CliError(msg)
    except urllib.error.URLError as e:
        raise CliError(
            f"cannot reach the gpu-queue daemon at {api_url()} ({e.reason}).\n"
            "Start it with: gq daemon start"
        )
    if raw:
        return resp
    return json.loads(resp.read())


def _get(path: str):
    return _request("GET", path)


def _post(path: str, body: dict = None):
    return _request("POST", path, body or {})


# -- formatting -----------------------------------------------------------


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h{m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d{h:02d}h"


def _job_runtime(job: dict) -> str:
    if job["started_at"] is None:
        return "-"
    end = job["finished_at"] or time.time()
    return _fmt_duration(end - job["started_at"])


def _print_table(rows, headers):
    if not rows:
        return
    widths = [max(len(str(r[i])) for r in rows + [headers]) for i in range(len(headers))]
    for row in [headers] + rows:
        print("  ".join(str(c).ljust(w) for c, w in zip(row, widths)).rstrip())


# -- commands --------------------------------------------------------------


def cmd_submit(args) -> int:
    command = list(args.command)
    gpus = args.gpus
    name = args.name
    workdir = os.getcwd()

    if not command:
        raise CliError("no command given (use `gq submit -- CMD ...` or `gq submit job.sh`)")

    # Script mode: first arg is an existing file that wasn't introduced by `--`
    script = Path(command[0])
    if not args.saw_dashdash and script.is_file():
        try:
            directives = parse_file(script)
        except JobScriptError as e:
            raise CliError(f"{script}: {e}")
        if gpus is None and "gpus" in directives:
            gpus = int(directives["gpus"])
        if name is None:
            name = directives.get("name")
        if "workdir" in directives:
            workdir = str(Path(directives["workdir"]).expanduser().resolve())
        command = ["/bin/bash", str(script.resolve())] + command[1:]
        if name is None:
            name = script.stem

    if gpus is None:
        gpus = 1

    env = {k: v for k, v in os.environ.items() if k not in _ENV_SKIP}
    job = _post(
        "/api/submit",
        {
            "name": name,
            "command": command,
            "workdir": workdir,
            "env": env,
            "gpus": gpus,
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
    )
    print(f"Submitted job {job['id']} ({job['name']}), {job['gpus_requested']} gpu(s)")
    return 0


def cmd_ls(args) -> int:
    if args.all:
        jobs = _get("/api/jobs")
    elif args.state:
        jobs = _get(f"/api/jobs?state={args.state.upper()}")
    else:
        active = _get("/api/jobs?state=QUEUED,RUNNING")
        recent = _get("/api/jobs?state=DONE,FAILED,CANCELLED&limit=5")
        # a job can finish between the two requests and show up in both
        jobs = list({j["id"]: j for j in active + recent}.values())
    jobs.sort(key=lambda j: j["id"])
    rows = []
    for j in jobs:
        gpu = ",".join(map(str, j["gpu_ids"])) if j["gpu_ids"] else f"({j['gpus_requested']} wanted)"
        exit_str = "" if j["exit_code"] is None else str(j["exit_code"])
        rows.append([j["id"], j["name"], j["state"], gpu, _job_runtime(j), exit_str,
                     j["conda_env"] or ""])
    _print_table(rows, ["ID", "NAME", "STATE", "GPUS", "TIME", "EXIT", "CONDA"])
    if not jobs:
        print("no jobs")
    return 0


def cmd_logs(args) -> int:
    offset = 0
    while True:
        resp = _request("GET", f"/api/jobs/{args.id}/logs?offset={offset}", raw=True)
        data = resp.read()
        state = resp.headers.get("X-Job-State", "")
        offset = int(resp.headers.get("X-Log-Offset", offset))
        if data:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        if not args.follow:
            if not data:
                return 0
            continue  # logs are chunked; keep draining until caught up
        if state not in ("QUEUED", "RUNNING") and not data:
            return 0  # job over and log drained
        time.sleep(1)


def cmd_cancel(args) -> int:
    for job_id in args.ids:
        job = _post(f"/api/jobs/{job_id}/cancel")
        print(f"job {job_id}: {job['state']}"
              + (" (SIGTERM sent)" if job["state"] == "RUNNING" else ""))
    return 0


def cmd_requeue(args) -> int:
    for job_id in args.ids:
        job = _post(f"/api/jobs/{job_id}/requeue")
        print(f"job {job_id}: {job['state']}")
    return 0


def cmd_gpus(args) -> int:
    gpus = _get("/api/gpus")
    rows = []
    for g in gpus:
        holder = f"job {g['job_id']}" if g["job_id"] is not None else ""
        rows.append([
            g["index"], g["name"], g["state"], holder,
            f"{g['mem_used_mb']}/{g['mem_total_mb']}MB", f"{g['util_pct']}%",
        ])
    _print_table(rows, ["GPU", "NAME", "STATE", "JOB", "MEM", "UTIL"])
    return 0


# -- daemon management ------------------------------------------------------


def _unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / SYSTEMD_UNIT


def _write_unit() -> None:
    gqd = Path(sys.argv[0]).resolve().parent / "gqd"
    if not gqd.exists():
        gqd = Path(sys.executable).parent / "gqd"
    unit = f"""[Unit]
Description=gpu-queue daemon (mini-Slurm for this machine)

[Service]
ExecStart={gqd}
Restart=on-failure
RestartSec=3
# Signal only the daemon on stop/restart — jobs live in the same cgroup and
# must survive daemon restarts (they're reconciled by pid on startup).
KillMode=process
TimeoutStopSec=15

[Install]
WantedBy=default.target
"""
    path = _unit_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(unit)


def _systemctl(*args) -> int:
    return subprocess.call(["systemctl", "--user", *args])


def cmd_daemon(args) -> int:
    if args.action == "run":
        from .daemon import main as daemon_main

        return daemon_main()
    if args.action == "start":
        _write_unit()
        if _systemctl("daemon-reload") != 0:
            raise CliError(
                "systemd user session unavailable; run the daemon directly with:\n"
                "  nohup gqd >> ~/.gpu-queue/daemon.log 2>&1 &"
            )
        if _systemctl("enable", "--now", SYSTEMD_UNIT) != 0:
            raise CliError(f"failed to start {SYSTEMD_UNIT}")
        print(f"daemon started ({SYSTEMD_UNIT}); state in {gq_home()}")
        return 0
    if args.action == "stop":
        if _unit_path().exists() and _systemctl("stop", SYSTEMD_UNIT) == 0:
            print("daemon stopped (running jobs keep running)")
            return 0
        _post("/api/shutdown")
        print("daemon asked to shut down (running jobs keep running)")
        return 0
    if args.action == "status":
        try:
            health = _get("/api/health")
            print(f"daemon running (version {health['version']}) at {api_url()}")
        except CliError as e:
            print(f"daemon NOT reachable: {e}")
            return 1
        return 0
    raise CliError(f"unknown daemon action {args.action!r}")


# -- entry point -------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="gq", description="mini-Slurm GPU job queue")
    p.add_argument("--version", action="version", version=f"gq {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("submit", help="submit a job (inline after --, or a job script)")
    sp.add_argument("--gpus", type=int, default=None, help="number of GPUs (default 1)")
    sp.add_argument("--name", default=None, help="job name")
    sp.add_argument("command", nargs=argparse.REMAINDER,
                    help="job script, or -- followed by the command")
    sp.set_defaults(fn=cmd_submit)

    lp = sub.add_parser("ls", help="list jobs (active + recently finished)")
    lp.add_argument("-a", "--all", action="store_true", help="show every job")
    lp.add_argument("--state", help="filter: QUEUED,RUNNING,DONE,FAILED,CANCELLED")
    lp.set_defaults(fn=cmd_ls)

    gp = sub.add_parser("logs", help="show a job's output")
    gp.add_argument("id", type=int)
    gp.add_argument("-f", "--follow", action="store_true", help="live-tail")
    gp.set_defaults(fn=cmd_logs)

    cp = sub.add_parser("cancel", help="cancel queued or running jobs")
    cp.add_argument("ids", type=int, nargs="+")
    cp.set_defaults(fn=cmd_cancel)

    rp = sub.add_parser("requeue", help="put finished jobs back in the queue")
    rp.add_argument("ids", type=int, nargs="+")
    rp.set_defaults(fn=cmd_requeue)

    up = sub.add_parser("gpus", help="show GPU allocation and external usage")
    up.set_defaults(fn=cmd_gpus)

    dp = sub.add_parser("daemon", help="manage the daemon")
    dp.add_argument("action", choices=["start", "stop", "status", "run"])
    dp.set_defaults(fn=cmd_daemon)
    return p


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv)
    # A `--` *leading* the remainder marks an inline command; a `--` later in
    # the argv belongs to the job (e.g. `gq submit job.sh -- --flag`).
    args.saw_dashdash = bool(getattr(args, "command", None)) and args.command[0] == "--"
    if args.saw_dashdash:
        args.command = args.command[1:]
    try:
        return args.fn(args)
    except CliError as e:
        print(f"gq: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
