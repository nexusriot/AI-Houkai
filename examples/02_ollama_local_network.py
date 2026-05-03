"""
AI-Houkai · Example 02 · Ollama on a local network


Connects to an Ollama instance running anywhere on your LAN
(a desktop, NAS, home server, etc.) and gives it persistent memory.

Typical setup
─────────────
  [Your laptop]  ──LAN──  [Home server 192.168.1.50]
       │                          │
  runs this script           ollama serve
  stores ChromaDB             (llama3.1 / mistral-nemo)

Start Ollama on the remote machine so it listens on all interfaces:
    OLLAMA_HOST=0.0.0.0 ollama serve

Pull a model that supports tool calling (once, on the remote machine):
    ollama pull llama3.1          # 8 B — good tool-call support
    ollama pull qwen2.5:7b        # lighter, also good
    ollama pull mistral-nemo      # 12 B, strong reasoning

Run this script (memory stays on YOUR machine):
    python examples/02_ollama_local_network.py

Override any default with env vars:
    OLLAMA_HOST=192.168.1.50  OLLAMA_PORT=11434  OLLAMA_MODEL=qwen2.5:7b \\
    AI_HOUKAI_PATH=/mnt/data/memory  python examples/02_ollama_local_network.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import urllib.request, urllib.error

try:
    from openai import OpenAI
except ImportError:
    sys.exit("pip install openai")

from ai_houkai.memory_system import MemoryStore


OLLAMA_HOST  = os.environ.get("OLLAMA_HOST",  "127.0.0.1")   # ← change me
OLLAMA_PORT  = int(os.environ.get("OLLAMA_PORT", "11434"))
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "nexusriot/gemma-4-abliterated:e2b")
MEMORY_PATH  = os.environ.get("AI_HOUKAI_PATH",
                               tempfile.mkdtemp(prefix="ai_houkai_ollama_lan_"))
CONNECT_TIMEOUT = int(os.environ.get("OLLAMA_TIMEOUT", "10"))   # seconds

OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/v1"



def _check_ollama() -> bool:
    """Return True if Ollama is reachable and the model is available."""

    url = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}/api/tags"
    try:
        with urllib.request.urlopen(url, timeout=CONNECT_TIMEOUT) as r:
            data = json.loads(r.read())
            models = [m["name"].split(":")[0] for m in data.get("models", [])]
            want   = OLLAMA_MODEL.split(":")[0]
            if want not in models:
                print(f"  ⚠  Model '{OLLAMA_MODEL}' not found on {OLLAMA_HOST}.")
                print(f"     Available: {', '.join(models) or '(none)'}")
                print(f"     Pull it:  ollama pull {OLLAMA_MODEL}")
                return False
            return True
    except urllib.error.URLError as exc:
        print(f"  ✗  Cannot reach Ollama at {OLLAMA_HOST}:{OLLAMA_PORT}  ({exc})")
        print(f"     Make sure Ollama is running with:")
        print(f"       OLLAMA_HOST=0.0.0.0 ollama serve")
        return False


store  = MemoryStore(path=MEMORY_PATH)
client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")

SYSTEM = """You are a helpful personal assistant with persistent long-term memory.

Memory tools available:
  • recall(query, k?, type?, min_importance?)  — search memories semantically
  • remember(text, type?, tags?, importance?)  — store a new memory
  • forget(memory_id)                          — delete a memory

Rules:
1. At the start of each reply, call `recall` with the user's topic.
2. If the user shares a preference, fact, decision, or task → call `remember`.
3. Never forget unless the user explicitly asks.
4. Be concise. Cite memory ids in square brackets when using stored facts."""

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "recall",
            "description": "Semantic search across stored memories.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query":          {"type": "string"},
                    "k":              {"type": "integer", "default": 5},
                    "type":           {"type": "string",
                                       "enum": ["episodic","semantic",
                                                "procedural","feedback"]},
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
            "description": "Persist a new memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text":       {"type": "string"},
                    "type":       {"type": "string",
                                   "enum": ["episodic","semantic",
                                            "procedural","feedback"],
                                   "default": "semantic"},
                    "tags":       {"type": "array", "items": {"type": "string"}},
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
            source=f"ollama/{OLLAMA_MODEL}",
        )
        return json.dumps({"id": mem.id, "stored": True})

    if name == "forget":
        return json.dumps({"deleted": store.forget(inputs["memory_id"])})

    return json.dumps({"error": f"unknown tool: {name}"})


def chat(history: list, user_message: str) -> str:
    history.append({"role": "user", "content": user_message})
    while True:
        resp = client.chat.completions.create(
            model=OLLAMA_MODEL,
            messages=history,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = resp.choices[0].message
        history.append(msg.model_dump())
        if not msg.tool_calls:
            return msg.content or ""
        for tc in msg.tool_calls:
            result = _dispatch_tool(tc.function.name, tc.function.arguments)
            print(f"  [tool] {tc.function.name}({tc.function.arguments[:70]})")
            history.append({"role": "tool",
                             "tool_call_id": tc.id,
                             "content": result})


# OLLAMA_MODEL = OLLAMA_MODEL   # keep module-level for _dispatch_tool source tag

def _print_banner() -> None:
    print(f"""
┌─────────────────────────────────────────────────────────┐
│  AI-Houkai  ×  Ollama LAN Agent                         │
│                                                         │
│  Model  : {OLLAMA_MODEL:<45} │
│  Host   : {OLLAMA_HOST}:{OLLAMA_PORT:<38} │
│  Memory : {MEMORY_PATH:<45} │
└─────────────────────────────────────────────────────────┘

Commands:
  memories  — list recent stored memories
  clear     — wipe ALL memories (asks for confirmation)
  quit      — exit
""")


def repl() -> None:
    _print_banner()

    if not _check_ollama():
        sys.exit(1)
    print(f"  ✓  Connected to {OLLAMA_HOST}:{OLLAMA_PORT}  model={OLLAMA_MODEL}\n")

    # Pre-seed some context so first conversation is useful out of the box
    if store.count() == 0:
        store.remember("User is a software developer building an AI agent.",
                       type="semantic", tags=["user-profile"], importance=0.8)
        store.remember("The project is called AI-Houkai and uses ChromaDB + Ollama.",
                       type="semantic", tags=["project"], importance=0.85)
        print("  (Seeded 2 bootstrap memories)\n")

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
            mems = store.list_recent(limit=15)
            if not mems:
                print("  (empty)\n")
            for m in mems:
                icon = {"semantic":"📘","procedural":"⚙️ ","episodic":"📅","feedback":"💬"}.get(m.type,"  ")
                print(f"  {icon} [{m.importance:.2f}] {m.text[:65]}")
            print()
            continue

        if user.lower() == "clear":
            confirm = input("  Delete ALL memories? (yes/no): ").strip().lower()
            if confirm == "yes":
                for m in store.list_recent(limit=1000):
                    store.forget(m.id)
                history = [{"role": "system", "content": SYSTEM}]
                print("  Cleared.\n")
            continue

        reply = chat(history, user)
        print(f"\nOllama: {reply}\n")


if __name__ == "__main__":
    repl()
