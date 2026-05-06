"""Resolve store path and collection from env, config file, or defaults."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

_DEFAULT_PATH = os.path.expanduser("~/.ai_houkai/.chroma")
_DEFAULT_COLLECTION = "ai_houkai"
_CONFIG_FILE = Path.home() / ".config" / "ai_houkai" / "config.toml"


@dataclass
class Config:
    store_path: str
    collection: str
    default_type: str
    default_importance: float
    editor: str


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
