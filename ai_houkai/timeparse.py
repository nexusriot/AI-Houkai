"""Lenient timestamp parsing shared by the CLI, MCP server and HTTP API.

`recall(since=…, until=…)` takes Unix timestamps (floats).  Humans and other
tools prefer ISO dates or relative spans, so the user-facing layers funnel
their inputs through :func:`parse_timestamp` to get a float the store accepts.
"""

from __future__ import annotations

import re
import time
from datetime import datetime, timezone

_REL_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*([smhdw])\s*$", re.IGNORECASE)
_REL_SECONDS = {"s": 1, "m": 60, "h": 3600, "d": 86_400, "w": 604_800}


def parse_timestamp(value: object, *, now: float | None = None) -> float | None:
    """Coerce *value* into a Unix timestamp (seconds since epoch).

    Accepts, in order:
      • ``None`` / empty string → ``None`` (no bound)
      • ``int`` / ``float`` → taken as an epoch timestamp verbatim
      • a relative span like ``"7d"``, ``"24h"``, ``"30m"`` → ``now`` minus that
        span (handy for "since 7 days ago")
      • a bare numeric string → parsed as an epoch timestamp
      • an ISO-8601 date/datetime (``"2026-06-14"`` or
        ``"2026-06-14T10:30:00"``) → its epoch timestamp; a trailing ``Z`` is
        honoured and naive values are interpreted as UTC

    Raises :class:`ValueError` for anything else so callers can surface a clear
    message instead of silently dropping the filter.
    """
    if value is None:
        return None
    if isinstance(value, bool):  # bool is an int subclass — reject explicitly
        raise ValueError(f"invalid timestamp: {value!r}")
    if isinstance(value, (int, float)):
        return float(value)

    text = str(value).strip()
    if not text:
        return None

    rel = _REL_RE.match(text)
    if rel:
        amount = float(rel.group(1))
        unit = rel.group(2).lower()
        base = time.time() if now is None else now
        return base - amount * _REL_SECONDS[unit]

    try:
        return float(text)
    except ValueError:
        pass

    iso = text[:-1] + "+00:00" if text.endswith("Z") else text
    try:
        dt = datetime.fromisoformat(iso)
    except ValueError as exc:
        raise ValueError(
            f"invalid timestamp {text!r}: expected epoch seconds, an ISO-8601 "
            f"date/datetime, or a relative span like '7d'"
        ) from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()
