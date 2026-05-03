"""Vector-backed memory store for an AI agent.

Uses ChromaDB as the persistence + similarity layer. Each memory is a
small document with structured metadata so the agent can filter as
well as semantically search.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

MemoryType = Literal["episodic", "semantic", "procedural", "feedback"]


@dataclass
class Memory:
    id: str
    text: str
    type: MemoryType
    tags: list[str] = field(default_factory=list)
    importance: float = 0.5  # 0..1
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    source: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "tags": ",".join(self.tags),
            "importance": self.importance,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "access_count": self.access_count,
            "source": self.source or "",
        }

    @classmethod
    def from_record(cls, mid: str, text: str, meta: dict[str, Any]) -> "Memory":
        return cls(
            id=mid,
            text=text,
            type=meta.get("type", "semantic"),
            tags=[t for t in meta.get("tags", "").split(",") if t],
            importance=float(meta.get("importance", 0.5)),
            created_at=float(meta.get("created_at", time.time())),
            last_accessed=float(meta.get("last_accessed", time.time())),
            access_count=int(meta.get("access_count", 0)),
            source=meta.get("source") or None,
        )


class MemoryStore:
    """A simple long-term memory backed by ChromaDB.

    The store keeps a single collection and uses metadata filters to
    separate memory kinds. We rely on Chroma's default sentence-
    transformers embedding so the prototype runs with no API keys.
    """

    def __init__(
        self,
        path: str = "./.chroma",
        collection: str = "ai_houkai",
        embedding_model: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.client = chromadb.PersistentClient(
            path=path, settings=Settings(anonymized_telemetry=False)
        )
        embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name=collection,
            embedding_function=embed_fn,
            metadata={"hnsw:space": "cosine"},
        )

    def remember(
        self,
        text: str,
        type: MemoryType = "semantic",
        tags: Iterable[str] = (),
        importance: float = 0.5,
        source: str | None = None,
    ) -> Memory:
        mem = Memory(
            id=str(uuid.uuid4()),
            text=text.strip(),
            type=type,
            tags=list(tags),
            importance=max(0.0, min(1.0, importance)),
            source=source,
        )
        self.collection.add(
            ids=[mem.id],
            documents=[mem.text],
            metadatas=[mem.to_metadata()],
        )
        return mem

    def forget(self, memory_id: str) -> bool:
        existing = self.collection.get(ids=[memory_id])
        if not existing["ids"]:
            return False
        self.collection.delete(ids=[memory_id])
        return True

    def recall(
        self,
        query: str,
        k: int = 5,
        type: MemoryType | None = None,
        tag: str | None = None,
        min_importance: float | None = None,
    ) -> list[tuple[Memory, float]]:
        if k <= 0 or self.collection.count() == 0:
            return []

        where: dict[str, Any] = {}
        if type:
            where["type"] = type
        if min_importance is not None:
            where["importance"] = {"$gte": min_importance}

        res = self.collection.query(
            query_texts=[query],
            n_results=k,
            where=where or None,
        )
        out: list[tuple[Memory, float]] = []
        for mid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            mem = Memory.from_record(mid, doc, meta)
            if tag and tag not in mem.tags:
                continue
            self._touch(mem)
            # cosine distance -> similarity
            out.append((mem, 1.0 - dist))
        return out

    def list_recent(self, limit: int = 20) -> list[Memory]:
        res = self.collection.get()
        memories = [
            Memory.from_record(i, d, m)
            for i, d, m in zip(res["ids"], res["documents"], res["metadatas"])
        ]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    def count(self) -> int:
        return self.collection.count()

    def _touch(self, mem: Memory) -> None:
        mem.last_accessed = time.time()
        mem.access_count += 1
        self.collection.update(ids=[mem.id], metadatas=[mem.to_metadata()])
