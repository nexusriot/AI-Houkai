"""Shared fixtures for memory system tests.

Uses a PersistentClient in a per-test tmp directory for full isolation.
EphemeralClient() reuses the same in-memory SQLite backend for every
call in the same process, causing count assertions to bleed across tests.

ChromaDB's PersistentClient opens file handles that persist until
explicitly closed. Without teardown this exhausts the OS file-descriptor
limit (~1024) after ~100 tests.
"""

from __future__ import annotations

import pytest

from ai_houkai.memory_system import MemoryStore


@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    """Fully isolated MemoryStore for each test (separate tmp dir)."""
    s = MemoryStore(path=str(tmp_path / "chroma"), collection="test_memory")
    yield s
    s.client.close()
