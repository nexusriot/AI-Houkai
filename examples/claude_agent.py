"""Claude agent with persistent memory via Anthropic tool use.

The agent exposes the memory store as three tools the model can call:
  • remember_tool  — store a new memory
  • recall_tool    — semantic search across stored memories
  • forget_tool    — delete a memory by id

How it works
------------
1. Every user message goes to Claude with the tool definitions.
2. Claude decides when to recall context or remember new facts.
3. Tool results are injected back as tool_result messages.
4. The loop continues until Claude stops calling tools.

Usage
-----
    export ANTHROPIC_API_KEY=sk-ant-...
    python examples/claude_agent.py

    # Or start a fresh session pointed at a specific chroma DB:
    AI_HOUKAI_PATH=/tmp/my_agent python examples/claude_agent.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import anthropic

# Allow running from repo root: python examples/claude_agent.py
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from memory_system import MemoryStore

MODEL = "claude-sonnet-4-6"
MEMORY_PATH = os.environ.get("AI_HOUKAI_PATH", tempfile.mkdtemp(prefix="ai_houkai_claude_"))

store = MemoryStore(path=MEMORY_PATH)
client = anthropic.Anthropic()

SYSTEM = """You are a helpful assistant with persistent long-term memory.

Use the memory tools proactively:
- Call `recall` at the start of every conversation to surface relevant context.
- Call `remember` when the user shares a preference, fact, or decision worth keeping.
- Call `forget` only when the user explicitly asks to remove something.

Be concise. Cite memory ids when you use stored facts."""

TOOLS: list[anthropic.types.ToolParam] = [
    {
        "name": "recall",
        "description": "Semantic search across stored memories. Returns the most relevant memories and their similarity scores.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "k": {"type": "integer", "default": 5, "description": "Max results"},
                "type": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural", "feedback"],
                    "description": "Optional: filter by memory type",
                },
                "min_importance": {
                    "type": "number",
                    "description": "Optional: only return memories with importance >= this value (0-1)",
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "remember",
        "description": "Store a new memory. Use for user preferences, decisions, facts, or anything worth recalling later.",
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The memory content"},
                "type": {
                    "type": "string",
                    "enum": ["episodic", "semantic", "procedural", "feedback"],
                    "default": "semantic",
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
    {
        "name": "forget",
        "description": "Delete a memory by its id.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {"type": "string", "description": "The memory id to delete"}
            },
            "required": ["memory_id"],
        },
    },
]


def _dispatch_tool(name: str, arguments: str) -> str:
    """Accepts a JSON string — same interface as openai/ollama agents."""
    inputs: dict = json.loads(arguments)

    if name == "recall":
        hits = store.recall(
            query=inputs["query"],
            k=inputs.get("k", 5),
            type=inputs.get("type"),
            min_importance=inputs.get("min_importance"),
        )
        if not hits:
            return json.dumps({"results": []})
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
            source="claude",
        )
        return json.dumps({"id": mem.id, "stored": True})

    if name == "forget":
        ok = store.forget(inputs["memory_id"])
        return json.dumps({"deleted": ok})

    return json.dumps({"error": f"unknown tool: {name}"})


def chat(history: list, user_message: str) -> str:
    history.append({"role": "user", "content": user_message})

    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM,
            tools=TOOLS,
            messages=history,
        )

        # Accumulate assistant turn (may contain text + tool calls)
        history.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            # Extract final text
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        # Process tool calls and feed results back
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                serialized = json.dumps(block.input, ensure_ascii=False)
                result = _dispatch_tool(block.name, serialized)
                print(f"  [tool] {block.name}({serialized[:80]})")
                tool_results.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    }
                )

        history.append({"role": "user", "content": tool_results})


def repl() -> None:
    print(f"Claude agent with memory  (db={MEMORY_PATH})")
    print("Type 'quit' to exit, 'memories' to list recent.\n")

    history: list = []

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
        print(f"Claude: {reply}\n")


if __name__ == "__main__":
    repl()
