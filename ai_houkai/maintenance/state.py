"""Persistence for maintenance run history.

Saved as JSON at ~/.ai_houkai/maintenance.state.json (configurable).
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class MaintenanceState:
    last_decay_at: float | None = None      # Unix timestamp of last decay run
    last_reflect_at: float | None = None    # Unix timestamp of last reflect run
    total_decayed: int = 0                  # cumulative memories pruned
    total_reflected: int = 0               # cumulative summaries created

    def save(self, path: str | Path) -> None:
        p = Path(path).expanduser()
        p.parent.mkdir(parents=True, exist_ok=True)
        # Write to a temp file in the same dir, then atomically rename, so a
        # crash mid-write can never leave a truncated/corrupt state file.
        tmp = p.with_name(f"{p.name}.{os.getpid()}.tmp")
        with open(tmp, "w") as f:
            json.dump(asdict(self), f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)

    @classmethod
    def load(cls, path: str | Path) -> "MaintenanceState":
        p = Path(path).expanduser()
        if not p.exists():
            return cls()
        # A corrupt/unreadable state file must not hard-stop the daemon: every
        # tick begins by loading state, so fall back to a fresh state instead.
        try:
            with open(p) as f:
                data: dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, OSError, ValueError):
            return cls()
        if not isinstance(data, dict):
            return cls()
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in data.items() if k in known})

    def next_run_at(self, last_at: float | None, interval: int, now: float | None = None) -> float:
        """Return the Unix timestamp when a job should next run.

        Pass the same ``now`` value used by the caller so there is no clock
        skew between the overdue check and the current time.
        """
        if last_at is None:
            t = now if now is not None else time.time()
            return t - 1                # never ran → immediately overdue
        return last_at + interval
