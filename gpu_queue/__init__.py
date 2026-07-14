"""gpu-queue: a mini-Slurm for a single shared GPU machine."""

import os
from pathlib import Path

__version__ = "0.1.0"

DEFAULT_PORT = 47563


def gq_home() -> Path:
    """State directory (~/.gpu-queue, overridable via GQ_HOME for tests)."""
    return Path(os.environ.get("GQ_HOME", str(Path.home() / ".gpu-queue")))


def log_dir() -> Path:
    return gq_home() / "logs"


def db_path() -> Path:
    return gq_home() / "db.sqlite"


def api_port() -> int:
    return int(os.environ.get("GQ_PORT", str(DEFAULT_PORT)))


def api_url() -> str:
    return f"http://127.0.0.1:{api_port()}"
