"""Persistence for maintenance run history.

Saved as JSON at ~/.ai_houkai/maintenance.state.json (configurable).
"""

from __future__ import annotations

import json
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
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "MaintenanceState":
        p = Path(path)
        if not p.exists():
            return cls()
        with open(p) as f:
            data: dict[str, Any] = json.load(f)
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
