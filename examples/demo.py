"""Quick walkthrough of the memory system without the MCP layer."""

from memory_system import MemoryStore


def main() -> None:
    store = MemoryStore(path="./.chroma_demo")

    store.remember(
        "User prefers terse answers without trailing summaries.",
        type="feedback",
        tags=["style", "tone"],
        importance=0.9,
    )
    store.remember(
        "Met with Anna on 2026-04-22 to scope the ingest rewrite.",
        type="episodic",
        tags=["meeting", "ingest"],
        importance=0.6,
    )
    store.remember(
        "ChromaDB cosine distance ranges 0..2; similarity = 1 - distance.",
        type="semantic",
        tags=["chromadb"],
        importance=0.4,
    )
    store.remember(
        "To deploy the API: run `make release` then `kubectl rollout restart`.",
        type="procedural",
        tags=["deploy", "api"],
        importance=0.7,
    )

    print(f"stored {store.count()} memories\n")

    print("== recall: 'how should I talk to the user?' ==")
    for mem, score in store.recall("how should I talk to the user?", k=3):
        print(f"  [{score:.3f}] ({mem.type}) {mem.text}")

    print("\n== recall procedural only: 'release process' ==")
    for mem, score in store.recall("release process", k=3, type="procedural"):
        print(f"  [{score:.3f}] {mem.text}")


if __name__ == "__main__":
    main()
