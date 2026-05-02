"""Shared fixtures for memory system tests.

Uses a PersistentClient in a per-test tmp directory for full isolation.
EphemeralClient() reuses the same in-memory SQLite backend for every
call in the same process, causing count assertions to bleed across tests.
"""

from __future__ import annotations

import pytest

from memory_system import MemoryStore


@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    """Fully isolated MemoryStore for each test (separate tmp dir)."""
    return MemoryStore(path=str(tmp_path / "chroma"), collection="test_memory")
