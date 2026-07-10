"""Resolve store path and collection from env, config file, or defaults."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

from ai_houkai.maintenance.durations import parse_duration

_DEFAULT_PATH = os.path.expanduser("~/.ai_houkai/.chroma")
_DEFAULT_COLLECTION = "ai_houkai"
_CONFIG_FILE = Path.home() / ".config" / "ai_houkai" / "config.toml"
_HOUKAI_DIR = Path.home() / ".ai_houkai"


@dataclass
class Config:
    store_path: str
    collection: str
    default_type: str
    default_importance: float | str   # a float, or "auto" → heuristic scorer
    editor: str


@dataclass
class MaintenanceConfig:
    enabled: bool
    decay_every: int | None         # seconds; None = disabled
    reflect_every: int | None       # seconds; None = disabled
    purge_every: int | None         # seconds; None = disabled (TTL reclamation)
    tick_interval: int              # seconds between loop wakes
    log_path: str
    state_path: str
    pid_path: str
    # DecayEngine params
    decay_rate: float
    min_score: float
    protect_types: tuple[str, ...]
    frequency_weight: float         # recall-reinforcement strength (0 = off)
    # ReflectionEngine params
    min_cluster_size: int
    reflect_apply: bool             # False → reflect in dry-run (observe only)
    summarizer: str | None          # e.g. "ollama:llama3.1"; None → extractive
    # ReflectionEngine.reflect(consolidate=…) value: False (none) leaves the
    # source episodics untouched, True (soft) supersedes them under the new
    # summary, "hard" deletes them. Default soft — without consolidation a
    # scheduled apply-mode reflection re-summarises the same clusters forever.
    reflect_consolidate: bool | str = True


def _resolve_importance(value: object) -> float | str:
    """A default_importance is a float, or the literal string "auto"."""
    if value == "auto":
        return "auto"
    return float(value)  # type: ignore[arg-type]


def _resolve_interval(value: object) -> int | None:
    """Parse a duration string/number/None/'off' to seconds."""
    if value is None or value == "off":
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return parse_duration(str(value))


def _resolve_consolidate(value: object) -> bool | str:
    """Map the [maintenance.reflect] consolidate knob to the engine's value."""
    if value in ("none", "off", False, None):
        return False
    if value in ("soft", True):
        return True
    if value == "hard":
        return "hard"
    raise ValueError(
        f"[maintenance.reflect] consolidate must be one of: none, soft, hard "
        f"— got {value!r}"
    )


def load() -> Config:
    file_cfg: dict = {}
    if _CONFIG_FILE.exists():
        with open(_CONFIG_FILE, "rb") as f:
            file_cfg = tomllib.load(f)

    return Config(
        # expanduser so `AI_HOUKAI_PATH=~/mem` / store_path = "~/mem" don't
        # create a literal ./~ directory (the shell only expands unquoted ~).
        store_path=os.path.expanduser(
            os.environ.get("AI_HOUKAI_PATH")
            or file_cfg.get("store_path", _DEFAULT_PATH)
        ),
        collection=os.environ.get("AI_HOUKAI_COLLECTION")
            or file_cfg.get("collection", _DEFAULT_COLLECTION),
        default_type=file_cfg.get("default_type", "semantic"),
        default_importance=_resolve_importance(
            file_cfg.get("default_importance", 0.5)
        ),
        editor=file_cfg.get("editor") or os.environ.get("EDITOR", "nano"),
    )


def load_maintenance() -> MaintenanceConfig:
    file_cfg: dict = {}
    if _CONFIG_FILE.exists():
        with open(_CONFIG_FILE, "rb") as f:
            file_cfg = tomllib.load(f)

    m = file_cfg.get("maintenance", {})
    decay_cfg = m.get("decay", {})
    reflect_cfg = m.get("reflect", {})

    return MaintenanceConfig(
        # Opt-out master switch: enabled = false disables every scheduled
        # maintenance surface (tick/run/start and the MCP maintenance_tick).
        enabled=bool(m.get("enabled", True)),
        decay_every=_resolve_interval(m.get("decay_every", "24h")),
        reflect_every=_resolve_interval(m.get("reflect_every", "7d")),
        purge_every=_resolve_interval(m.get("purge_every", "24h")),
        tick_interval=_resolve_interval(m.get("tick_interval", "5m")) or 300,
        log_path=os.path.expanduser(
            str(m.get("log_path", str(_HOUKAI_DIR / "maintenance.log")))
        ),
        state_path=os.path.expanduser(
            str(m.get("state_path", str(_HOUKAI_DIR / "maintenance.state.json")))
        ),
        pid_path=os.path.expanduser(
            str(m.get("pid_path", str(_HOUKAI_DIR / "maintenance.pid")))
        ),
        decay_rate=float(decay_cfg.get("decay_rate", 0.1)),
        min_score=float(decay_cfg.get("min_score", 0.05)),
        protect_types=tuple(decay_cfg.get("protect_types", ["procedural"])),
        frequency_weight=float(decay_cfg.get("frequency_weight", 0.0)),
        min_cluster_size=int(reflect_cfg.get("min_cluster_size", 3)),
        reflect_apply=bool(reflect_cfg.get("apply", False)),
        summarizer=os.environ.get("AI_HOUKAI_SUMMARIZER")
            or reflect_cfg.get("summarizer") or None,
        reflect_consolidate=_resolve_consolidate(
            reflect_cfg.get("consolidate", "soft")
        ),
    )
