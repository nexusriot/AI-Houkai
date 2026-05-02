"""OpenAI agent with persistent memory via function calling.

The agent exposes the memory store as three functions the model calls:
  • recall    — semantic search across stored memories
  • remember  — store a new memory
  • forget    — delete a memory by id

How it works
------------
1. Every user message goes to the model with the tool definitions.
2. The model decides when to recall context or store new facts.
3. Tool results are fed back as "tool" role messages.
4. The loop continues until the model stops calling tools.

Supports any OpenAI chat model with function-calling:
  gpt-4o, gpt-4o-mini, gpt-4-turbo, gpt-3.5-turbo (≥ 1106)

Usage
-----
    export OPENAI_API_KEY=sk-...
    python examples/openai_agent.py

    # Override model or memory path:
    OPENAI_MODEL=gpt-4o-mini AI_HOUKAI_PATH=/tmp/oai python examples/openai_agent.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_system import MemoryStore

MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MEMORY_PATH = os.environ.get(
    "AI_HOUKAI_PATH", tempfile.mkdtemp(prefix="ai_houkai_openai_")
)

store = MemoryStore(path=MEMORY_PATH)
client = OpenAI()  # reads OPENAI_API_KEY from env

SYSTEM = """You are a helpful assistant with persistent long-term memory.

Use the memory tools proactively:
- Call `recall` at the start of every reply to surface relevant context.
- Call `remember` when the user shares a preference, fact, or decision worth keeping.
- Call `forget` only when the user explicitly asks to remove something.

Be concise. When using stored facts, cite the memory id."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": (
                "Semantic search across stored memories. "
                "Returns the most relevant memories with similarity scores."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Natural language search query",
                    },
                    "k": {
                        "type": "integer",
                        "default": 5,
                        "description": "Max number of results to return",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "procedural", "feedback"],
                        "description": "Optional: restrict results to a single memory type",
                    },
                    "min_importance": {
                        "type": "number",
                        "description": "Optional: only return memories with importance >= this (0-1)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": (
                "Store a new memory. Use for user preferences, decisions, "
                "facts, or anything worth recalling later."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The memory content",
                    },
                    "type": {
                        "type": "string",
                        "enum": ["episodic", "semantic", "procedural", "feedback"],
                        "default": "semantic",
                        "description": "Memory category",
                    },
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional topical tags",
                    },
                    "importance": {
                        "type": "number",
                        "default": 0.5,
                        "description": "Importance 0-1. Use >0.8 for critical preferences.",
                    },
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Permanently delete a memory by its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "string",
                        "description": "The id of the memory to delete",
                    }
                },
                "required": ["memory_id"],
            },
        },
    },
]


def _dispatch_tool(name: str, arguments: str) -> str:
    inputs: dict = json.loads(arguments)

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
            source="openai",
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

        # Store assistant message (OpenAI SDK object → plain dict for re-use)
        history.append(
            {
                "role": "assistant",
                "content": msg.content,
                "tool_calls": (
                    [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        for tc in msg.tool_calls
                    ]
                    if msg.tool_calls
                    else None
                ),
            }
        )

        if not msg.tool_calls:
            return msg.content or ""

        # Execute every tool call and collect results
        for tc in msg.tool_calls:
            result = _dispatch_tool(tc.function.name, tc.function.arguments)
            short_args = tc.function.arguments[:80]
            print(f"  [tool] {tc.function.name}({short_args})")
            history.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )


def repl() -> None:
    print(f"OpenAI agent ({MODEL}) with memory  (db={MEMORY_PATH})")
    print("Type 'quit' to exit, 'memories' to list recent.\n")

    history: list = [{"role": "system", "content": SYSTEM}]

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
                print(
                    f"  [{m.type}] {m.text[:70]}  "
                    f"(importance={m.importance}, accesses={m.access_count})"
                )
            continue

        reply = chat(history, user)
        print(f"GPT: {reply}\n")


if __name__ == "__main__":
    repl()
