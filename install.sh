#!/usr/bin/env bash
# gpu-queue installer — no prerequisites beyond python3 (and curl/git for a
# remote install). Creates a private venv so gq never depends on, or pollutes,
# whatever conda env you happen to have active.
#
#   curl -fsSL https://raw.githubusercontent.com/Fernando961226/gpu-queue/main/install.sh | bash
#   ./install.sh --with-daemon     # also enable the systemd user service
#   ./install.sh --dev             # editable install + test deps (dev machines)
#   ./install.sh --uninstall
set -euo pipefail

REPO_URL="${GQ_REPO_URL:-https://github.com/Fernando961226/gpu-queue.git}"
PREFIX="${GQ_PREFIX:-$HOME/.local/share/gpu-queue}"
BIN_DIR="${GQ_BIN_DIR:-$HOME/.local/bin}"
VENV="$PREFIX/venv"
COMMANDS=(gq gqd)

WITH_DAEMON=0
UNINSTALL=0
DEV=0
for arg in "$@"; do
    case "$arg" in
        --with-daemon) WITH_DAEMON=1 ;;
        --uninstall)   UNINSTALL=1 ;;
        --dev)         DEV=1 ;;
        -h|--help)     sed -n '2,8p' "$0" | sed 's/^# \?//'; exit 0 ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

say()  { printf '\033[1m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33mwarning:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31merror:\033[0m %s\n' "$*" >&2; exit 1; }

# -- uninstall ---------------------------------------------------------------

if [ "$UNINSTALL" -eq 1 ]; then
    if command -v systemctl >/dev/null 2>&1; then
        systemctl --user disable --now gpu-queue.service 2>/dev/null || true
    fi
    rm -f "$HOME/.config/systemd/user/gpu-queue.service"
    for cmd in "${COMMANDS[@]}"; do
        # only remove symlinks that point into our prefix
        target="$(readlink -f "$BIN_DIR/$cmd" 2>/dev/null || true)"
        case "$target" in "$PREFIX"/*) rm -f "$BIN_DIR/$cmd" ;; esac
    done
    rm -rf "$PREFIX"
    say "uninstalled. State in ~/.gpu-queue (jobs, logs) was left alone."
    exit 0
fi

# -- locate the source -------------------------------------------------------

# Installing from a checkout (./install.sh) uses that tree; a piped curl
# install has no checkout, so fall back to the public repo.
SCRIPT_DIR=""
if [ -n "${BASH_SOURCE[0]:-}" ] && [ -f "${BASH_SOURCE[0]}" ]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

if [ -n "${GQ_SOURCE:-}" ]; then
    SOURCE="$GQ_SOURCE"
elif [ -n "$SCRIPT_DIR" ] && [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    SOURCE="$SCRIPT_DIR"
    say "installing from local checkout: $SOURCE"
else
    command -v git >/dev/null 2>&1 || die "git is required to install from $REPO_URL"
    SOURCE="git+$REPO_URL"
    say "installing from $REPO_URL"
fi

# -- pick a python that can actually build a venv ----------------------------

# Needs >=3.9 and a working venv module. On Debian/Ubuntu the stdlib python3
# often ships without ensurepip (python3-venv is a separate, sudo-requiring
# package), so we probe candidates by trying for real rather than trusting
# `import venv`, and fall back to conda's python, which always works.
pick_python() {
    local candidates=() py v
    [ -n "${GQ_PYTHON:-}" ] && candidates+=("$GQ_PYTHON")
    # System interpreters first, by absolute path: a bare `python3.12` would
    # resolve to whatever conda env is active, and the daemon must not depend
    # on conda staying installed or healthy.
    for v in 3.13 3.12 3.11 3.10 3.9; do
        candidates+=("/usr/bin/python$v")
    done
    candidates+=(/usr/bin/python3 /usr/local/bin/python3)
    # Only then fall back to whatever is on PATH (likely conda).
    candidates+=(python3)
    [ -n "${CONDA_PREFIX:-}" ] && candidates+=("$CONDA_PREFIX/bin/python3")
    candidates+=("$HOME/miniconda3/bin/python3" "$HOME/anaconda3/bin/python3")

    local probe
    probe="$(mktemp -d)"
    trap 'rm -rf "$probe"' RETURN

    for py in "${candidates[@]}"; do
        command -v "$py" >/dev/null 2>&1 || continue
        "$py" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)' 2>/dev/null || continue
        if "$py" -m venv --without-pip "$probe/v" >/dev/null 2>&1; then
            rm -rf "$probe/v"
            printf '%s' "$(command -v "$py")"
            return 0
        fi
    done
    return 1
}

PYTHON="$(pick_python)" || die "no usable python3 (need >=3.9 with the venv module).
Try: sudo apt install python3-venv   — or set GQ_PYTHON=/path/to/python3"
say "using python: $PYTHON ($("$PYTHON" -c 'import platform; print(platform.python_version())'))"

# -- build the venv and install ----------------------------------------------

mkdir -p "$PREFIX" "$BIN_DIR"

if [ -x "$VENV/bin/python" ]; then
    say "reusing existing venv at $VENV (upgrading in place)"
else
    say "creating venv at $VENV"
    "$PYTHON" -m venv "$VENV" || die "failed to create venv at $VENV"
fi

# --dev keeps the venv pointing at the checkout, so edits take effect without
# reinstalling, and pulls in pytest.
PIP_ARGS=()
if [ "$DEV" -eq 1 ]; then
    case "$SOURCE" in
        git+*) die "--dev needs a local checkout; clone the repo and run ./install.sh --dev" ;;
    esac
    PIP_ARGS=(--editable "$SOURCE[dev]")
    say "dev mode: editable install from $SOURCE"
else
    PIP_ARGS=("$SOURCE")
fi

# Reinstall unconditionally for remote installs. `--upgrade` alone compares
# version numbers, so re-running this after a change that did not bump the
# version leaves the old code in place and the upgrade silently does nothing.
# An editable install already points at the checkout, so it needs no such thing.
FORCE=()
if [ "$DEV" -eq 0 ]; then
    FORCE=(--force-reinstall --no-deps)
fi

# uv is 10-100x faster when it happens to be present; never required.
if command -v uv >/dev/null 2>&1; then
    say "installing gpu-queue (uv)"
    uv pip install --quiet --python "$VENV/bin/python" --upgrade "${FORCE[@]}" "${PIP_ARGS[@]}"
    [ "$DEV" -eq 0 ] && uv pip install --quiet --python "$VENV/bin/python" "${PIP_ARGS[@]}"
else
    say "installing gpu-queue (pip)"
    "$VENV/bin/python" -m pip install --quiet --upgrade pip >/dev/null
    "$VENV/bin/python" -m pip install --quiet --upgrade "${FORCE[@]}" "${PIP_ARGS[@]}"
    # --no-deps above skips dependency resolution, so make sure they are present.
    [ "$DEV" -eq 0 ] && "$VENV/bin/python" -m pip install --quiet "${PIP_ARGS[@]}"
fi

# -- expose the commands -----------------------------------------------------

for cmd in "${COMMANDS[@]}"; do
    [ -x "$VENV/bin/$cmd" ] || die "$cmd missing from the venv — install failed"
    existing="$(readlink -f "$BIN_DIR/$cmd" 2>/dev/null || true)"
    if [ -e "$BIN_DIR/$cmd" ] && [ ! -L "$BIN_DIR/$cmd" ]; then
        die "$BIN_DIR/$cmd exists and is not a symlink; move it aside and re-run"
    fi
    ln -sfn "$VENV/bin/$cmd" "$BIN_DIR/$cmd"
done
say "linked ${COMMANDS[*]} into $BIN_DIR"

# A gq earlier on PATH (e.g. a stale `pip install` into conda base) would win.
resolved="$(command -v gq 2>/dev/null || true)"
case ":$PATH:" in
    *":$BIN_DIR:"*) ;;
    *) warn "$BIN_DIR is not on your PATH. Add to your shell rc:
    export PATH=\"$BIN_DIR:\$PATH\"" ;;
esac
if [ -n "$resolved" ] && [ "$resolved" != "$BIN_DIR/gq" ]; then
    warn "another gq is earlier on PATH and will shadow this one: $resolved"
fi

# -- daemon ------------------------------------------------------------------

if [ "$WITH_DAEMON" -eq 1 ]; then
    say "starting the daemon"
    "$BIN_DIR/gq" daemon start
else
    cat <<EOF

Installed. Next:

  gq daemon start          # enable the systemd user service
  gq submit --gpus 1 -- python train.py
  gq ls

EOF
fi
