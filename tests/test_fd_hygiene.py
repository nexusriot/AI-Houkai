"""Regression guard for the ChromaDB file-descriptor leak.

A full `pytest tests/` run creates hundreds of stores. Any that are never
closed (CLI invocations under CliRunner, forgotten fixture teardowns) keep
their SQLite/HNSW handles open via Chroma's process-global System registry.
Before the `_release_chroma_systems` autouse fixture in conftest.py, those
handles accumulated until the run blew past the OS fd limit (~1024 in the
test container) and the last tests to run failed at setup with
`chromadb.errors.InternalError: unable to open database file (code 14)`.

This test reproduces the leak in miniature and asserts that the same sweep
the autouse teardown performs — stopping the still-cached Systems — reclaims
the descriptors.
"""

from __future__ import annotations

import os

import pytest

from ai_houkai.memory_system import MemoryStore

# Exercise the exact reclamation the autouse fixture uses.
from .conftest import _release_chroma_fds


def _open_fds() -> int:
    return len(os.listdir(f"/proc/{os.getpid()}/fd"))


@pytest.mark.skipif(
    not os.path.isdir("/proc/self/fd"),
    reason="fd counting relies on Linux /proc",
)
def test_unclosed_stores_are_reclaimed(tmp_path):
    # Warm up: the first store load pulls in the embedding model / torch
    # runtime, which open their own long-lived fds. Do that before measuring
    # so the baseline already includes them and we isolate the per-store cost.
    warm = MemoryStore(path=str(tmp_path / "warm"), collection="warmcol")
    warm.remember("warm up the embedding model")
    warm.client.close()
    del warm
    _release_chroma_fds()
    baseline = _open_fds()

    n = 25
    stores = [
        MemoryStore(path=str(tmp_path / f"s{i}"), collection="leakcol")
        for i in range(n)
    ]
    # Sanity: unclosed stores really do hold descriptors — several each, so
    # far more than one-per-store.
    grew = _open_fds() - baseline
    assert grew >= n, f"expected open stores to hold fds, only grew +{grew}"

    # Drop our references and run the teardown sweep. The still-cached Systems
    # are stopped, closing their files immediately (no gc needed).
    stores.clear()
    _release_chroma_fds()

    # The sweep must reclaim the vast majority of the leak. A regressed fixture
    # (no reclamation) would leave ~`grew` descriptors open; we require the
    # residual well under that, with slack for pytest's own fd churn.
    residual = _open_fds() - baseline
    assert residual <= grew // 3, (
        f"sweep reclaimed only {grew - residual} of {grew} leaked fds "
        f"(residual +{residual})"
    )
