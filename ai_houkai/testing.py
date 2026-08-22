"""Test helpers: a dependency-free embedder and pytest fixtures.

The default store loads a real sentence-transformers model, which makes the
test suite take minutes and forces every contributor (and every CI job) to have
torch installed. :class:`FakeEmbedder` hashes text into a deterministic unit
vector instead, so a store can be built in microseconds::

    from ai_houkai.memory_system import MemoryStore
    from ai_houkai.testing import FakeEmbedder

    store = MemoryStore(path=tmp, embedding_function=FakeEmbedder())

Or, with the pytest plugin registered (see below), just take the fixture::

    def test_something(fake_store):
        fake_store.remember("hello")

The vectors are meaningless as *semantics* — hashing does not preserve meaning,
so "cat" and "kitten" are as far apart as "cat" and "tectonics". That is fine
for testing plumbing (writes, links, filters, journal, packing, surfaces) and
wrong for testing ranking *quality*; use a real model for that, and prefer
:mod:`ai_houkai.eval` with a gold set to measure it.

Determinism is the point: the same text always yields the same vector, in this
process and the next, so tests that assert on ordering stay stable.

To get the fixtures, add to ``conftest.py``::

    pytest_plugins = ["ai_houkai.testing"]
"""

from __future__ import annotations

import hashlib
import math
import struct
from typing import Iterator, Sequence

import pytest

from ai_houkai.memory_system import MemoryStore

__all__ = ["FakeEmbedder", "fake_embedder", "fake_store", "make_store"]

_DEFAULT_DIM = 32


class FakeEmbedder:
    """Deterministic hash-based embedding function.

    Expands a BLAKE2b digest of the text into ``dim`` floats and L2-normalises
    them, so cosine similarity stays in [-1, 1] and identical texts embed
    identically. No model, no network, no torch.
    """

    def __init__(self, dim: int = _DEFAULT_DIM) -> None:
        if dim <= 0:
            raise ValueError("dim must be > 0")
        self.dim = dim

    def name(self) -> str:
        return f"ai_houkai.fake:{self.dim}"

    def _vector(self, text: str) -> list[float]:
        # Each 8-byte block of the digest stream yields one float in [-1, 1);
        # re-hash with a counter to reach any dim without truncating entropy.
        raw = bytearray()
        counter = 0
        while len(raw) < self.dim * 8:
            h = hashlib.blake2b(f"{counter}\x00{text}".encode("utf-8"), digest_size=64)
            raw.extend(h.digest())
            counter += 1
        vals = [
            (struct.unpack_from("<Q", raw, i * 8)[0] / 2**63) - 1.0
            for i in range(self.dim)
        ]
        norm = math.sqrt(sum(v * v for v in vals))
        if norm == 0:  # astronomically unlikely, but a zero vector breaks cosine
            vals[0] = 1.0
            norm = 1.0
        return [v / norm for v in vals]

    def __call__(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002
        return [self._vector(t) for t in input]


def make_store(path, *, collection: str = "test", dim: int = _DEFAULT_DIM,
               **kwargs) -> MemoryStore:
    """Build a MemoryStore backed by :class:`FakeEmbedder` at *path*.

    Extra keyword arguments go straight to :class:`MemoryStore`, so a test can
    still set ``conflict_policy``, ``journal_enabled``, ``importance_fn``, etc.
    """
    return MemoryStore(
        path=str(path),
        collection=collection,
        embedding_function=FakeEmbedder(dim),
        **kwargs,
    )


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    """A deterministic embedding function for direct injection."""
    return FakeEmbedder()


@pytest.fixture()
def fake_store(tmp_path) -> Iterator[MemoryStore]:
    """A throwaway MemoryStore that needs no embedding model.

    Closes the Chroma client on teardown so Windows/CI runners don't leak file
    handles between tests.
    """
    store = make_store(tmp_path / "chroma")
    try:
        yield store
    finally:
        store.client.close()
