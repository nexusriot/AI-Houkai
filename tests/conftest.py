"""Shared fixtures for memory system tests.

Uses a PersistentClient in a per-test tmp directory for full isolation.
EphemeralClient() reuses the same in-memory SQLite backend for every
call in the same process, causing count assertions to bleed across tests.

ChromaDB's PersistentClient opens file handles (SQLite + HNSW) that live
until the client is closed *or* its System is evicted from Chroma's
process-global registry. The fixtures below close what they open, but
several code paths under test legitimately do not — most notably every
in-process ``CliRunner.invoke()`` builds a store in the Typer callback and
relies on process exit (which never happens in-process) to reclaim it.
Left alone, those handles accumulate across a full run and exhaust the OS
file-descriptor limit (~1024 in the test container); whichever tests run
last then fail at setup with SQLite "unable to open database file"
(code 14). ``_release_chroma_systems`` (autouse) stops any never-closed
Systems and clears that registry after every test, giving each test the
clean-slate reclamation a fresh process would get. See
tests/test_fd_hygiene.py for the regression guard.
"""

from __future__ import annotations

import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from ai_houkai.memory_system import MemoryStore


def _release_chroma_fds() -> None:
    """Stop every ChromaDB System still cached in the process-global registry.

    A closed client removes its System from the registry (``close()`` →
    ``_release_system`` pops it), so what remains here is exactly the stores
    that were *never* closed. Stopping each one closes its SQLite/HNSW files
    immediately and deterministically — no ``gc.collect()``, which would
    otherwise cost ~130ms per call walking the loaded torch/model heap
    (~200s across the suite). Then clear the (now-dead) registry entries.
    """
    for system in list(getattr(SharedSystemClient, "_identifier_to_system", {}).values()):
        try:
            system.stop()
        except Exception:
            pass
    SharedSystemClient.clear_system_cache()


@pytest.fixture(autouse=True)
def _release_chroma_systems():
    """Reclaim ChromaDB file descriptors after every test.

    Chroma caches each client's System (holding the SQLite connection and
    HNSW file handles) in a process-global registry keyed by settings.
    Stores that are never explicitly closed — CLI invocations under
    ``CliRunner``, and any fixture/test that forgets to close — otherwise
    leak those handles until the run exhausts the fd limit. Runs for
    chroma-free tests too, where it is a cheap no-op.

    Teardown order guarantees this runs *after* fixtures like ``store`` have
    already called ``client.close()``, so it only sweeps up what was left
    unclosed — it never yanks a handle out from under a live client.
    """
    yield
    _release_chroma_fds()


@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    """Fully isolated MemoryStore for each test (separate tmp dir)."""
    s = MemoryStore(path=str(tmp_path / "chroma"), collection="test_memory")
    yield s
    s.client.close()
