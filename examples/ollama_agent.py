"""Ollama local agent with persistent memory.

Uses Ollama's OpenAI-compatible tool-call API so the agent can
call `recall`, `remember`, and `forget` the same way the Claude
example does — but 100% local, no API keys needed.

Prerequisites
-------------
    # Install Ollama: https://ollama.com
    ollama pull llama3.1        # 8B, supports tool use
    # or
    ollama pull mistral-nemo    # lighter alternative
    pip install openai          # Ollama exposes an OAI-compat endpoint

Usage
-----
    OLLAMA_MODEL=llama3.1 python examples/ollama_agent.py
    # Or just:
    python examples/ollama_agent.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_system import MemoryStore

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")
MEMORY_PATH = os.environ.get("AI_HOUKAI_PATH", tempfile.mkdtemp(prefix="ai_houkai_ollama_"))

store = MemoryStore(path=MEMORY_PATH)
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")  # key unused by Ollama

SYSTEM = """You are a helpful assistant with persistent long-term memory.

Use the memory tools proactively:
- Call `recall` at the start of every reply to surface relevant context.
- Call `remember` when the user shares a preference, fact, or decision.
- Call `forget` only when the user explicitly asks to remove something.

Be concise."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Semantic search across stored memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "k": {"type": "integer", "default": 5},
                    "type": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "procedural", "feedback"],
                    },
                    "min_importance": {"type": "number"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store a new memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "type": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "procedural", "feedback"],
                        "default": "semantic",
                    },
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "importance": {"type": "number", "default": 0.5},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Delete a memory by id.",
            "parameters": {
                "type": "object",
                "properties": {"memory_id": {"type": "string"}},
                "required": ["memory_id"],
            },
        },
    },
]


def _dispatch_tool(name: str, arguments: str) -> str:
    inputs = json.loads(arguments)

    if name == "recall":
        hits = store.recall(
            query=inputs["query"],
            k=inputs.get("k", 5),
            type=inputs.get("type"),
            min_importance=inputs.get("min_importance"),
        )
        return json.dumps(
            {
                "results": [
                    {
                        "id": m.id,
                        "text": m.text,
                        "type": m.type,
                        "tags": m.tags,
                        "importance": m.importance,
                        "score": round(score, 4),
                    }
                    for m, score in hits
                ]
            }
        )

    if name == "remember":
        mem = store.remember(
            text=inputs["text"],
            type=inputs.get("type", "semantic"),
            tags=inputs.get("tags", []),
            importance=inputs.get("importance", 0.5),
            source="ollama",
        )
        return json.dumps({"id": mem.id, "stored": True})

    if name == "forget":
        ok = store.forget(inputs["memory_id"])
        return json.dumps({"deleted": ok})

    return json.dumps({"error": f"unknown tool: {name}"})


def chat(history: list, user_message: str) -> str:
    history.append({"role": "user", "content": user_message})

    while True:
        response = client.chat.completions.create(
            model=MODEL,
            messages=history,
            tools=TOOLS,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        history.append(msg.model_dump())  # keep full message for context

        if not msg.tool_calls:
            return msg.content or ""

        # Process every tool call in the response
        for tc in msg.tool_calls:
            result = _dispatch_tool(tc.function.name, tc.function.arguments)
            print(f"  [tool] {tc.function.name}({tc.function.arguments[:80]})")
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )


def repl() -> None:
    print(f"Ollama agent ({MODEL}) with memory  (db={MEMORY_PATH})")
    print("Type 'quit' to exit, 'memories' to list recent.\n")

    # Pre-seed some example memories so the demo works out of the box
    if store.count() == 0:
        store.remember(
            "User is a Python developer working on an AI agent project.",
            type="semantic",
            tags=["user-profile"],
            importance=0.8,
        )
        store.remember(
            "Project uses ChromaDB for vector storage and Ollama for local inference.",
            type="semantic",
            tags=["project", "tech-stack"],
            importance=0.85,
        )
        print("  (Pre-seeded 2 demo memories)\n")

    history = [{"role": "system", "content": SYSTEM}]

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user:
            continue
        if user.lower() == "quit":
            break
        if user.lower() == "memories":
            recent = store.list_recent(limit=10)
            if not recent:
                print("  (no memories stored yet)")
            for m in recent:
                print(f"  [{m.type}] {m.text[:70]}  (importance={m.importance})")
            continue

        reply = chat(history, user)
        print(f"Ollama: {reply}\n")


if __name__ == "__main__":
    repl()
