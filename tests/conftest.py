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

import os

import pytest
from chromadb.api.shared_system_client import SharedSystemClient

from ai_houkai.memory_system import MemoryStore
from ai_houkai.memory_system import store as store_module
from ai_houkai.testing import FakeEmbedder

# Exposes the fake_store / fake_embedder fixtures shipped for downstream users
# in ai_houkai.testing, so this suite dogfoods the same helpers.
pytest_plugins = ["ai_houkai.testing"]


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


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "needs_model: requires the real sentence-transformers embedder — the "
        "assertion depends on actual semantic similarity, which the hash-based "
        "FakeEmbedder cannot provide. Deselected in the fast CI job.",
    )


@pytest.fixture(autouse=True)
def _default_to_fake_embedder(request, monkeypatch):
    """Swap the default local model for FakeEmbedder in most tests.

    Patching ``local_embedder`` — the single place the store resolves its
    default — covers *every* store built during a test, not just the ones
    using the ``store`` fixture: the MCP server's lazy store, CLI runs under
    CliRunner, HTTP fixtures, and direct constructions all inherit it.

    Tests marked ``needs_model`` opt back into the real sentence-transformers
    model because their assertions depend on genuine semantic similarity
    (conflict thresholds, reflection clustering, ranking quality).
    ``AI_HOUKAI_TEST_REAL_EMBEDDER=1`` forces the real model everywhere, which
    is how that marker set was derived — re-run with it to re-derive.
    """
    if os.environ.get("AI_HOUKAI_TEST_REAL_EMBEDDER") == "1":
        return
    if request.node.get_closest_marker("needs_model") is not None:
        return
    monkeypatch.setattr(store_module, "local_embedder",
                        lambda *_args, **_kwargs: FakeEmbedder())


@pytest.fixture()
def store(tmp_path) -> MemoryStore:
    """Fully isolated MemoryStore for each test (separate tmp dir)."""
    s = MemoryStore(path=str(tmp_path / "chroma"), collection="test_memory")
    yield s
    s.client.close()
