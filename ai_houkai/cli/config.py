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
    default_importance: float
    editor: str


@dataclass
class MaintenanceConfig:
    enabled: bool
    decay_every: int | None         # seconds; None = disabled
    reflect_every: int | None       # seconds; None = disabled
    tick_interval: int              # seconds between loop wakes
    log_path: str
    state_path: str
    pid_path: str
    # DecayEngine params
    decay_rate: float
    min_score: float
    protect_types: tuple[str, ...]
    # ReflectionEngine params
    min_cluster_size: int
    reflect_apply: bool             # False → reflect in dry-run (observe only)


def _resolve_interval(value: object) -> int | None:
    """Parse a duration string/number/None/'off' to seconds."""
    if value is None or value == "off":
        return None
    if isinstance(value, (int, float)):
        return int(value)
    return parse_duration(str(value))


def load() -> Config:
    file_cfg: dict = {}
    if _CONFIG_FILE.exists():
        with open(_CONFIG_FILE, "rb") as f:
            file_cfg = tomllib.load(f)

    return Config(
        store_path=os.environ.get("AI_HOUKAI_PATH")
            or file_cfg.get("store_path", _DEFAULT_PATH),
        collection=os.environ.get("AI_HOUKAI_COLLECTION")
            or file_cfg.get("collection", _DEFAULT_COLLECTION),
        default_type=file_cfg.get("default_type", "semantic"),
        default_importance=float(file_cfg.get("default_importance", 0.5)),
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
        enabled=bool(m.get("enabled", False)),
        decay_every=_resolve_interval(m.get("decay_every", "24h")),
        reflect_every=_resolve_interval(m.get("reflect_every", "7d")),
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
        min_cluster_size=int(reflect_cfg.get("min_cluster_size", 3)),
        reflect_apply=bool(reflect_cfg.get("apply", False)),
    )
