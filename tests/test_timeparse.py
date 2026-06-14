"""Tests for the lenient timestamp parser shared by CLI / MCP / HTTP."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from ai_houkai.timeparse import parse_timestamp


class TestParseTimestamp:
    def test_none_and_empty(self):
        assert parse_timestamp(None) is None
        assert parse_timestamp("") is None
        assert parse_timestamp("   ") is None

    def test_epoch_passthrough(self):
        assert parse_timestamp(1718323200.0) == 1718323200.0
        assert parse_timestamp(1718323200) == 1718323200.0

    def test_numeric_string(self):
        assert parse_timestamp("1718323200") == 1718323200.0

    def test_relative_spans(self):
        now = 1_000_000.0
        assert parse_timestamp("7d", now=now) == now - 7 * 86_400
        assert parse_timestamp("24h", now=now) == now - 24 * 3600
        assert parse_timestamp("30m", now=now) == now - 30 * 60
        assert parse_timestamp("2w", now=now) == now - 2 * 604_800
        assert parse_timestamp("10s", now=now) == now - 10

    def test_iso_date_is_utc(self):
        ts = parse_timestamp("2026-06-14")
        expected = datetime(2026, 6, 14, tzinfo=timezone.utc).timestamp()
        assert ts == expected

    def test_iso_datetime_with_z(self):
        ts = parse_timestamp("2026-06-14T12:00:00Z")
        expected = datetime(2026, 6, 14, 12, tzinfo=timezone.utc).timestamp()
        assert ts == expected

    def test_bool_rejected(self):
        with pytest.raises(ValueError):
            parse_timestamp(True)

    def test_garbage_rejected(self):
        with pytest.raises(ValueError):
            parse_timestamp("not-a-date")
