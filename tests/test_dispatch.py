"""Tests that the _dispatch_tool helper (shared logic across all agent
examples) produces correct JSON for recall / remember / forget.

We monkey-patch the global `store` in each example module so tests
run without any API keys or live services.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import types

import pytest

from memory_system import MemoryStore

_EXAMPLES_DIR = os.path.join(os.path.dirname(__file__), "..", "examples")

AGENTS = [
    "claude_agent.py",
    "04_openai.py",
    "02_ollama_local_network.py",
]


def _load_dispatch(filename: str, store: MemoryStore):
    """Load an agent example module by file path and inject our test store."""
    # Derive a safe module name (digits-prefixed filenames are invalid identifiers)
    module_name = "_agent_" + filename.replace(".py", "").replace("-", "_")

    # Evict any previously loaded version
    if module_name in sys.modules:
        del sys.modules[module_name]

    # Stub SDK client constructors so no network calls happen at import time.
    fake_client = types.SimpleNamespace(
        chat=types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kw: None)
        ),
        messages=types.SimpleNamespace(create=lambda **kw: None),
    )
    sys.modules.setdefault("openai", types.ModuleType("openai"))
    sys.modules["openai"].OpenAI = lambda **kw: fake_client  # type: ignore
    sys.modules.setdefault("anthropic", types.ModuleType("anthropic"))
    sys.modules["anthropic"].Anthropic = lambda: fake_client  # type: ignore

    path = os.path.join(_EXAMPLES_DIR, filename)
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)  # type: ignore

    mod.store = store  # inject isolated test store
    return mod._dispatch_tool


@pytest.fixture(params=AGENTS)
def dispatch(request, tmp_path):
    """Fresh isolated store + dispatch fn for each test × agent combination."""
    store = MemoryStore(path=str(tmp_path / "chroma"), collection="dispatch_test")
    fn = _load_dispatch(request.param, store)
    return fn, store


class TestDispatchRemember:
    def test_returns_id_and_stored_true(self, dispatch):
        fn, _ = dispatch
        result = json.loads(fn("remember", json.dumps({"text": "Python is great"})))
        assert result["stored"] is True
        assert "id" in result

    def test_with_type_and_tags(self, dispatch):
        fn, store = dispatch
        result = json.loads(
            fn(
                "remember",
                json.dumps(
                    {
                        "text": "Run make release to deploy",
                        "type": "procedural",
                        "tags": ["deploy"],
                        "importance": 0.9,
                    }
                ),
            )
        )
        assert result["stored"] is True
        assert store.count() == 1


class TestDispatchRecall:
    def _seed(self, store: MemoryStore):
        store.remember("Python is great for scripting", type="semantic", importance=0.7)
        store.remember("Deploy with make release", type="procedural", importance=0.9)

    def test_returns_results_list(self, dispatch):
        fn, store = dispatch
        self._seed(store)
        result = json.loads(fn("recall", json.dumps({"query": "Python scripting"})))
        assert "results" in result
        assert isinstance(result["results"], list)

    def test_result_has_expected_keys(self, dispatch):
        fn, store = dispatch
        self._seed(store)
        result = json.loads(fn("recall", json.dumps({"query": "deploy"})))
        if result["results"]:
            r = result["results"][0]
            for key in ("id", "text", "type", "tags", "importance", "score"):
                assert key in r

    def test_empty_store_returns_empty_list(self, dispatch):
        fn, _ = dispatch
        # Store is fresh — tmp_path gives a new dir per test, zero documents.
        result = json.loads(fn("recall", json.dumps({"query": "anything", "k": 5})))
        assert result["results"] == []


class TestDispatchForget:
    def test_deletes_existing_memory(self, dispatch):
        fn, store = dispatch
        mem = store.remember("to be deleted")
        assert store.count() == 1
        result = json.loads(fn("forget", json.dumps({"memory_id": mem.id})))
        assert result["deleted"] is True
        assert store.count() == 0

    def test_unknown_id_returns_false(self, dispatch):
        fn, _ = dispatch
        result = json.loads(
            fn("forget", json.dumps({"memory_id": "00000000-0000-0000-0000-000000000000"}))
        )
        assert result["deleted"] is False


class TestDispatchUnknownTool:
    def test_returns_error(self, dispatch):
        fn, _ = dispatch
        result = json.loads(fn("nonexistent_tool", json.dumps({})))
        assert "error" in result
