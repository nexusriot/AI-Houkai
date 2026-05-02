"""
AI-Houkai · Example 04 · OpenAI with persistent memory

A personal coding assistant that remembers your preferences,
your project details, and past decisions across sessions.

Supported models (any with function-calling):
  gpt-4o          — best reasoning, higher cost
  gpt-4o-mini     — fast, cheap, good enough for most tasks  ← default
  gpt-4-turbo     — legacy but solid

Usage
─────
    export OPENAI_API_KEY=sk-...
    python examples/04_openai.py

    # Persist memory across sessions:
    AI_HOUKAI_PATH=~/.ai_houkai python examples/04_openai.py

    # Use a different model:
    OPENAI_MODEL=gpt-4o python examples/04_openai.py

REPL commands
─────────────
  memories          — list stored memories with metadata
  search <query>    — raw semantic search, shows scores
  forget <id>       — delete a memory by its id
  clear             — wipe all memories (asks for confirmation)
  quit / exit       — leave
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")

from memory_system import MemoryStore

MODEL       = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
MEMORY_PATH = os.environ.get("AI_HOUKAI_PATH",
                              tempfile.mkdtemp(prefix="ai_houkai_openai_"))

store  = MemoryStore(path=MEMORY_PATH)
client = OpenAI()   # reads OPENAI_API_KEY from env


SYSTEM = """You are a senior software-engineering assistant with persistent memory.

Memory tools:
  recall(query, k?, type?, min_importance?)  — semantic search
  remember(text, type?, tags?, importance?)  — store something
  forget(memory_id)                          — delete by id

Workflow on every turn:
1. Call recall() with the user's topic — always, even for short questions.
2. Scan results for relevant context before composing your answer.
3. If the user reveals a preference, corrects something, or makes a
   decision → call remember() immediately, importance ≥ 0.8 for
   preferences, ≥ 0.9 for critical facts.

memory types:
  semantic   → facts, tech knowledge
  procedural → how-to steps, workflows
  episodic   → past events, decisions made
  feedback   → user preferences, style notes

Format rules (inferred from feedback memories at runtime):
  Apply any style preferences found in feedback memories.
  Default: concise bullets, no fluff, code where relevant."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Semantic search across all stored memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":          {"type": "string"},
                    "k":              {"type": "integer", "default": 5},
                    "type":           {"type": "string",
                                       "enum": ["episodic", "semantic",
                                                "procedural", "feedback"]},
                    "min_importance": {"type": "number",
                                       "description": "0-1 threshold"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "remember",
            "description": "Store a new memory for future recall.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":       {"type": "string"},
                    "type":       {"type": "string",
                                   "enum": ["episodic", "semantic",
                                            "procedural", "feedback"],
                                   "default": "semantic"},
                    "tags":       {"type": "array",
                                   "items": {"type": "string"}},
                    "importance": {"type": "number",
                                   "default": 0.5,
                                   "description": "0-1; use 0.9+ for critical facts"},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "forget",
            "description": "Delete a memory by its id.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {"type": "string"},
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
        return json.dumps({
            "results": [
                {"id": m.id, "text": m.text, "type": m.type,
                 "tags": m.tags, "importance": m.importance,
                 "score": round(s, 4)}
                for m, s in hits
            ]
        })

    if name == "remember":
        mem = store.remember(
            text=inputs["text"],
            type=inputs.get("type", "semantic"),
            tags=inputs.get("tags", []),
            importance=inputs.get("importance", 0.5),
            source=f"openai/{MODEL}",
        )
        return json.dumps({"id": mem.id, "stored": True})

    if name == "forget":
        return json.dumps({"deleted": store.forget(inputs["memory_id"])})

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

        history.append({
            "role": "assistant",
            "content": msg.content,
            "tool_calls": (
                [{"id": tc.id, "type": "function",
                  "function": {"name": tc.function.name,
                               "arguments": tc.function.arguments}}
                 for tc in msg.tool_calls]
                if msg.tool_calls else None
            ),
        })

        if not msg.tool_calls:
            return msg.content or ""

        for tc in msg.tool_calls:
            result = _dispatch_tool(tc.function.name, tc.function.arguments)
            print(f"  \033[90m[{tc.function.name}] {tc.function.arguments[:72]}\033[0m")
            history.append({"role": "tool",
                             "tool_call_id": tc.id,
                             "content": result})


ICONS = {"semantic": "📘", "procedural": "⚙️ ", "episodic": "📅", "feedback": "💬"}

def _cmd_memories() -> None:
    mems = store.list_recent(limit=20)
    if not mems:
        print("  (no memories stored)\n")
        return
    print(f"\n  {'TYPE':11s}  {'IMP':4s}  {'ACC':3s}  TEXT")
    print(f"  {'─'*11}  {'─'*4}  {'─'*3}  {'─'*50}")
    for m in mems:
        icon = ICONS.get(m.type, "  ")
        print(f"  {icon} {m.type:9s}  {m.importance:.2f}  {m.access_count:3d}  "
              f"{m.text[:60]}")
    print()

def _cmd_search(query: str) -> None:
    hits = store.recall(query, k=8)
    if not hits:
        print("  (no results)\n")
        return
    print(f"\n  {'SCORE':5s}  {'TYPE':11s}  TEXT")
    for mem, score in hits:
        print(f"  {score:.3f}  {mem.type:11s}  {mem.text[:60]}")
    print()

def _cmd_forget(mid: str) -> None:
    if store.forget(mid):
        print(f"  Deleted {mid}\n")
    else:
        print(f"  Not found: {mid}\n")

def _cmd_clear(history: list) -> None:
    confirm = input("  Delete ALL memories? (yes/no): ").strip().lower()
    if confirm == "yes":
        deleted = 0
        for m in store.list_recent(limit=9999):
            store.forget(m.id)
            deleted += 1
        history.clear()
        history.append({"role": "system", "content": SYSTEM})
        print(f"  Cleared {deleted} memories.\n")


def _banner() -> None:
    print(f"""
┌─────────────────────────────────────────────────────────┐
│  AI-Houkai  ×  OpenAI Coding Assistant                  │
│                                                         │
│  Model  : {MODEL:<45} │
│  Memory : {MEMORY_PATH:<45} │
│  Count  : {store.count()} stored memories{' ' * (35 - len(str(store.count())))} │
└─────────────────────────────────────────────────────────┘

  Type  'memories'        to browse stored memories
        'search <query>'  for a raw semantic search
        'forget <id>'     to delete a memory
        'clear'           to wipe all memories
        'quit'            to exit
""")


def main() -> None:
    _banner()

    # Seed a helpful starter context if the store is empty
    if store.count() == 0:
        store.remember(
            "New session started. No prior context yet.",
            type="episodic", tags=["session"], importance=0.3,
        )

    history: list = [{"role": "system", "content": SYSTEM}]

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not user:
            continue

        low = user.lower()

        if low in ("quit", "exit"):
            print("Bye.")
            break
        elif low == "memories":
            _cmd_memories()
        elif low.startswith("search "):
            _cmd_search(user[7:].strip())
        elif low.startswith("forget "):
            _cmd_forget(user[7:].strip())
        elif low == "clear":
            _cmd_clear(history)
        else:
            reply = chat(history, user)
            print(f"\nGPT: {reply}\n")


if __name__ == "__main__":
    main()
