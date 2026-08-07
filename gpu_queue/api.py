"""Localhost HTTP API served from inside the daemon.

Endpoints (all JSON unless noted):
    GET  /api/health
    GET  /api/jobs?state=RUNNING,QUEUED&limit=50
    GET  /api/jobs/<id>
    GET  /api/jobs/<id>/logs?offset=N     (text/plain; X-Log-Offset header
                                           carries the next offset for tailing)
    GET  /api/gpus
    POST /api/submit        {name, command, workdir, env, gpus, conda_env}
    POST /api/jobs/<id>/cancel
    POST /api/jobs/<id>/requeue
    POST /api/shutdown
"""

import json
import logging
import re
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from . import __version__, api_port

MAX_LOG_CHUNK = 512 * 1024

log = logging.getLogger("gqd.api")


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class Handler(BaseHTTPRequestHandler):
    daemon_obj = None  # set by serve()
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # quiet access log
        pass

    # -- helpers -------------------------------------------------------------

    def _json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return {}
        try:
            return json.loads(self.rfile.read(length))
        except json.JSONDecodeError:
            raise ApiError(400, "invalid JSON body")

    def _job_or_404(self, job_id: int):
        job = self.daemon_obj.db.get_job(job_id)
        if job is None:
            raise ApiError(404, f"no job {job_id}")
        return job

    # -- routing -------------------------------------------------------------

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def _route(self, method: str):
        try:
            url = urlparse(self.path)
            q = parse_qs(url.query)
            path = url.path.rstrip("/")
            m = re.fullmatch(r"/api/jobs/(\d+)(/logs|/cancel|/requeue)?", path)
            if method == "GET" and path == "/api/health":
                self._json({"ok": True, "version": __version__})
            elif method == "GET" and path == "/api/jobs":
                self._list_jobs(q)
            elif method == "GET" and m and m.group(2) is None:
                self._json(self._job_or_404(int(m.group(1))).to_dict())
            elif method == "GET" and m and m.group(2) == "/logs":
                self._logs(int(m.group(1)), q)
            elif method == "GET" and path == "/api/gpus":
                self._json(self.daemon_obj.gpu_status())
            elif method == "POST" and path == "/api/submit":
                self._submit()
            elif method == "POST" and m and m.group(2) == "/cancel":
                self._json(self.daemon_obj.cancel(int(m.group(1))).to_dict())
            elif method == "POST" and m and m.group(2) == "/requeue":
                self._json(self.daemon_obj.requeue(int(m.group(1))).to_dict())
            elif method == "POST" and path == "/api/shutdown":
                self._json({"ok": True})
                self.daemon_obj.request_shutdown()
            else:
                raise ApiError(404, f"unknown endpoint {method} {path}")
        except ApiError as e:
            self._json({"error": str(e)}, status=e.status)
        except KeyError as e:
            self._json({"error": f"no job {e.args[0]}"}, status=404)
        except ValueError as e:
            self._json({"error": str(e)}, status=400)
        except BrokenPipeError:
            pass
        except Exception as e:  # don't let one request kill the thread
            # A 500 means a bug in the daemon, so keep the traceback: log it
            # here (journalctl --user -u gpu-queue) and return it too. Leaking
            # internals is normally a bad idea, but this API is bound to
            # localhost for a single user, and without it every crash reads as
            # a bare "internal error" with no file or line to go on.
            log.exception("unhandled error in %s %s", method, self.path)
            self._json(
                {"error": f"internal error: {e}", "traceback": traceback.format_exc()},
                status=500,
            )

    # -- endpoints -----------------------------------------------------------

    def _list_jobs(self, q):
        states = None
        if "state" in q:
            states = [s.strip().upper() for s in q["state"][0].split(",") if s.strip()]
        limit = int(q["limit"][0]) if "limit" in q else None
        jobs = self.daemon_obj.db.list_jobs(states=states, limit=limit)
        self._json([j.to_dict() for j in jobs])

    def _submit(self):
        body = self._read_body()
        for key in ("command", "workdir", "env"):
            if key not in body:
                raise ApiError(400, f"missing field {key!r}")
        if not body["command"]:
            raise ApiError(400, "empty command")
        job = self.daemon_obj.submit(
            name=body.get("name") or _default_name(body["command"]),
            command=body["command"],
            workdir=body["workdir"],
            env=body["env"],
            gpus=int(body.get("gpus", 1)),
            conda_env=body.get("conda_env"),
            vram_mb=body.get("vram_mb"),
        )
        self._json(job.to_dict(), status=201)

    def _logs(self, job_id: int, q):
        job = self._job_or_404(job_id)
        offset = int(q["offset"][0]) if "offset" in q else 0
        data = b""
        size = offset
        if job.log_path:
            try:
                with open(job.log_path, "rb") as fh:
                    fh.seek(0, 2)
                    size = fh.tell()
                    if offset < size:
                        fh.seek(offset)
                        data = fh.read(MAX_LOG_CHUNK)
            except FileNotFoundError:
                pass
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Log-Offset", str(offset + len(data)))
        self.send_header("X-Job-State", job.state)
        self.end_headers()
        self.wfile.write(data)


def _default_name(command) -> str:
    for word in command:
        base = word.rsplit("/", 1)[-1]
        if base not in ("python", "python3", "bash", "sh", "env"):
            return base[:40]
    return command[0].rsplit("/", 1)[-1][:40]


def serve(daemon_obj) -> ThreadingHTTPServer:
    """Start the API server in a background thread; returns the server."""
    handler = type("BoundHandler", (Handler,), {"daemon_obj": daemon_obj})
    server = ThreadingHTTPServer(("127.0.0.1", api_port()), handler)
    t = threading.Thread(target=server.serve_forever, name="gq-api", daemon=True)
    t.start()
    return server
