"""Parse and format human-readable duration strings.

Examples
    parse_duration("30m")  →  1800
    parse_duration("24h")  →  86400
    parse_duration("7d")   →  604800
    format_duration(3661)  →  "1h 1m"
"""

from __future__ import annotations

import re

_UNITS: dict[str, int] = {
    "s": 1,
    "m": 60,
    "h": 3_600,
    "d": 86_400,
    "w": 604_800,
}

_PATTERN = re.compile(r"^(\d+(?:\.\d+)?)([smhdw])$")


def parse_duration(s: str) -> int:
    """Parse a duration string into seconds.  Raises ValueError on bad input.

    Supported units: s (seconds), m (minutes), h (hours), d (days), w (weeks).
    The string "off" is not accepted here — callers should handle it before calling.
    """
    s = s.strip().lower()
    m = _PATTERN.match(s)
    if not m:
        raise ValueError(
            f"Cannot parse duration {s!r}. "
            "Expected <number><unit> where unit ∈ {{s, m, h, d, w}} "
            "(e.g. '30m', '24h', '7d')."
        )
    value, unit = float(m.group(1)), m.group(2)
    return int(value * _UNITS[unit])


def format_duration(seconds: float) -> str:
    """Format a number of seconds as a compact human-readable string."""
    secs = int(abs(seconds))
    if secs < 60:
        return f"{secs}s"
    if secs < 3_600:
        m, s = divmod(secs, 60)
        return f"{m}m {s}s" if s else f"{m}m"
    if secs < 86_400:
        h, rem = divmod(secs, 3_600)
        mins = rem // 60
        return f"{h}h {mins}m" if mins else f"{h}h"
    d, rem = divmod(secs, 86_400)
    h = rem // 3_600
    return f"{d}d {h}h" if h else f"{d}d"
