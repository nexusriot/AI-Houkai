"""PID file management and detached-process helpers for the maintenance daemon.

We spawn the daemon as a detached subprocess (using start_new_session=True)
rather than a double-fork, which is safer with open file handles from
ChromaDB and avoids fork-safety issues on Linux.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
from pathlib import Path


def get_pid(pid_path: str | Path) -> int | None:
    """Read the daemon's PID from the pid file.  Returns None if not found."""
    p = Path(pid_path)
    if not p.exists():
        return None
    try:
        return int(p.read_text().strip())
    except (ValueError, OSError):
        return None


def write_pid(pid_path: str | Path, pid: int) -> None:
    p = Path(pid_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(str(pid))


def remove_pid(pid_path: str | Path) -> None:
    Path(pid_path).unlink(missing_ok=True)


def is_alive(pid_path: str | Path) -> bool:
    """Return True if a process with the recorded PID is running."""
    pid = get_pid(pid_path)
    if pid is None:
        return False
    try:
        os.kill(pid, 0)     # signal 0 = existence check, no signal sent
        return True
    except (ProcessLookupError, PermissionError):
        return False


def stop_daemon(pid_path: str | Path) -> bool:
    """Send SIGTERM to the daemon.  Returns True if the signal was sent."""
    pid = get_pid(pid_path)
    if pid is None:
        return False
    try:
        os.kill(pid, signal.SIGTERM)
        return True
    except ProcessLookupError:
        remove_pid(pid_path)
        return False


def spawn_detached(
    store_path: str,
    collection: str,
    log_path: str,
    pid_path: str,
) -> int:
    """Spawn 'houkai maintenance run' as a detached background process.

    Uses start_new_session=True so the child survives the parent's exit.
    Returns the child's PID.
    """
    log_p = Path(log_path)
    log_p.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "ai_houkai.cli",
        "--store", store_path,
        "--collection", collection,
        "maintenance", "run",
    ]

    log_fd = open(log_p, "a")
    proc = subprocess.Popen(
        cmd,
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=log_fd,
        stderr=log_fd,
        close_fds=True,
    )
    write_pid(pid_path, proc.pid)
    return proc.pid
