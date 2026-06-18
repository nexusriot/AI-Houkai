"""Tests for ai_houkai.memory_system.summarizers — spec parsing, providers,
fallback behaviour, and wiring into the scheduler/config."""

from __future__ import annotations

import json
import sys
import types

import pytest

from ai_houkai.memory_system.reflection import _default_summarizer
from ai_houkai.memory_system.store import Memory
from ai_houkai.memory_system import summarizers
from ai_houkai.memory_system.summarizers import build_summarizer, render_prompt


def _mems() -> list[Memory]:
    return [
        Memory(id="a", text="fixed the deploy script", type="episodic",
               importance=0.4),
        Memory(id="b", text="deploys go through make release", type="episodic",
               importance=0.9),
    ]



class TestSpecParsing:
    def test_none_returns_extractive_default(self):
        assert build_summarizer(None) is _default_summarizer
        assert build_summarizer("") is _default_summarizer

    @pytest.mark.parametrize("spec", ["extractive", "default", "none"])
    def test_builtin_aliases(self, spec):
        assert build_summarizer(spec) is _default_summarizer

    def test_extractive_with_model_part_is_still_extractive(self):
        assert build_summarizer("extractive:whatever") is _default_summarizer

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown summarizer provider"):
            build_summarizer("gemini:flash")

    def test_missing_model_raises(self):
        with pytest.raises(ValueError, match="missing a model"):
            build_summarizer("ollama")
        with pytest.raises(ValueError, match="missing a model"):
            build_summarizer("ollama:")

    def test_model_may_contain_colons(self, monkeypatch):
        seen = {}

        def fake_urlopen(req, timeout=0):
            seen["body"] = json.loads(req.data.decode())
            return _fake_response("ok")

        monkeypatch.setattr(summarizers.urllib.request, "urlopen", fake_urlopen)
        build_summarizer("ollama:llama3.1:8b")(_mems())
        assert seen["body"]["model"] == "llama3.1:8b"



class TestPrompt:
    def test_prompt_contains_all_texts_importance_first(self):
        prompt = render_prompt(_mems())
        assert "- deploys go through make release" in prompt
        assert "- fixed the deploy script" in prompt
        assert prompt.index("make release") < prompt.index("deploy script")



class _FakeHTTPResponse:
    def __init__(self, payload: dict):
        self._body = json.dumps(payload).encode()

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _fake_response(content: str) -> _FakeHTTPResponse:
    return _FakeHTTPResponse(
        {"choices": [{"message": {"content": content}}]}
    )


class TestOllama:
    def test_calls_openai_compat_endpoint(self, monkeypatch):
        seen = {}

        def fake_urlopen(req, timeout=0):
            seen["url"] = req.full_url
            seen["body"] = json.loads(req.data.decode())
            return _fake_response("Deploys are done via make release.")

        monkeypatch.setattr(summarizers.urllib.request, "urlopen", fake_urlopen)
        monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)

        out = build_summarizer("ollama:llama3.1")(_mems())

        assert out == "Deploys are done via make release."
        assert seen["url"] == "http://localhost:11434/v1/chat/completions"
        assert seen["body"]["model"] == "llama3.1"
        assert "make release" in seen["body"]["messages"][0]["content"]

    def test_base_url_env_with_or_without_v1(self, monkeypatch):
        seen = {}

        def fake_urlopen(req, timeout=0):
            seen["url"] = req.full_url
            return _fake_response("x")

        monkeypatch.setattr(summarizers.urllib.request, "urlopen", fake_urlopen)

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://box:11434/v1")
        build_summarizer("ollama:m")(_mems())
        assert seen["url"] == "http://box:11434/v1/chat/completions"

        monkeypatch.setenv("OLLAMA_BASE_URL", "http://box:11434/")
        build_summarizer("ollama:m")(_mems())
        assert seen["url"] == "http://box:11434/v1/chat/completions"



class TestFallback:
    def test_falls_back_to_extractive_on_error(self, monkeypatch):
        def boom(req, timeout=0):
            raise OSError("connection refused")

        monkeypatch.setattr(summarizers.urllib.request, "urlopen", boom)
        out = build_summarizer("ollama:llama3.1")(_mems())
        assert out == _default_summarizer(_mems())

    def test_falls_back_on_empty_output(self, monkeypatch):
        monkeypatch.setattr(
            summarizers.urllib.request, "urlopen",
            lambda req, timeout=0: _fake_response("   "),
        )
        out = build_summarizer("ollama:llama3.1")(_mems())
        assert out == _default_summarizer(_mems())

    def test_no_fallback_raises(self, monkeypatch):
        def boom(req, timeout=0):
            raise OSError("connection refused")

        monkeypatch.setattr(summarizers.urllib.request, "urlopen", boom)
        with pytest.raises(OSError):
            build_summarizer("ollama:llama3.1", fallback=False)(_mems())

    def test_output_is_stripped(self, monkeypatch):
        monkeypatch.setattr(
            summarizers.urllib.request, "urlopen",
            lambda req, timeout=0: _fake_response("  summary text\n"),
        )
        assert build_summarizer("ollama:m")(_mems()) == "summary text"

    def test_openai_client_built_lazily_and_falls_back(self, monkeypatch):
        # Regression: OpenAI() was constructed inside build_summarizer(), so a
        # missing/invalid key crashed the build (and the unattended maintenance
        # daemon) before _with_fallback could catch it. The client must be
        # built at call time so the failure degrades to extractive instead.
        built: list[bool] = []

        class _BoomClient:
            def __init__(self, **kw):
                built.append(True)
                raise RuntimeError("Missing credentials")

        monkeypatch.setattr(summarizers, "OpenAI", _BoomClient)

        summ = build_summarizer("openai:gpt-4o-mini")   # must NOT raise
        assert built == []                              # not built at build time
        assert summ(_mems()) == _default_summarizer(_mems())  # degrades at call
        assert built == [True]                          # built once, at call time

    def test_anthropic_client_built_lazily_and_falls_back(self, monkeypatch):
        class _BoomClient:
            def __init__(self, **kw):
                raise RuntimeError("Missing credentials")

        monkeypatch.setattr(summarizers, "Anthropic", _BoomClient)

        summ = build_summarizer("anthropic:claude-haiku-4-5")   # must NOT raise
        assert summ(_mems()) == _default_summarizer(_mems())

    def test_construction_error_surfaces_at_call_without_fallback(self, monkeypatch):
        class _BoomClient:
            def __init__(self, **kw):
                raise RuntimeError("Missing credentials")

        monkeypatch.setattr(summarizers, "OpenAI", _BoomClient)

        summ = build_summarizer("openai:gpt-4o-mini", fallback=False)  # build ok
        with pytest.raises(RuntimeError, match="Missing credentials"):
            summ(_mems())                               # raised lazily, at call



class TestSdkProviders:
    def test_openai_provider(self, monkeypatch):
        seen = {}

        def create(**kw):
            seen.update(kw)
            msg = types.SimpleNamespace(content="openai summary")
            return types.SimpleNamespace(
                choices=[types.SimpleNamespace(message=msg)]
            )

        client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create))
        )
        monkeypatch.setattr(summarizers, "OpenAI", lambda **kw: client)

        out = build_summarizer("openai:gpt-4o-mini")(_mems())
        assert out == "openai summary"
        assert seen["model"] == "gpt-4o-mini"
        assert "make release" in seen["messages"][0]["content"]

    def test_anthropic_provider(self, monkeypatch):
        seen = {}

        def create(**kw):
            seen.update(kw)
            block = types.SimpleNamespace(type="text", text="anthropic summary")
            return types.SimpleNamespace(content=[block])

        client = types.SimpleNamespace(
            messages=types.SimpleNamespace(create=create)
        )
        monkeypatch.setattr(summarizers, "Anthropic", lambda **kw: client)

        out = build_summarizer("anthropic:claude-haiku-4-5")(_mems())
        assert out == "anthropic summary"
        assert seen["model"] == "claude-haiku-4-5"
        assert seen["max_tokens"] > 0

    def test_missing_sdk_raises_import_error_with_hint(self, monkeypatch):
        monkeypatch.setattr(summarizers, "OpenAI", None)
        with pytest.raises(ImportError, match="ai-houkai\\[openai\\]"):
            build_summarizer("openai:gpt-4o-mini")



class TestWiring:
    def test_env_var_overrides_config(self, monkeypatch, tmp_path):
        from ai_houkai.cli import config as cfgmod

        toml = tmp_path / "config.toml"
        toml.write_text(
            "[maintenance.reflect]\nsummarizer = \"ollama:from-file\"\n"
        )
        monkeypatch.setattr(cfgmod, "_CONFIG_FILE", toml)

        monkeypatch.delenv("AI_HOUKAI_SUMMARIZER", raising=False)
        assert cfgmod.load_maintenance().summarizer == "ollama:from-file"

        monkeypatch.setenv("AI_HOUKAI_SUMMARIZER", "ollama:from-env")
        assert cfgmod.load_maintenance().summarizer == "ollama:from-env"

    def test_unset_summarizer_is_none(self, monkeypatch, tmp_path):
        from ai_houkai.cli import config as cfgmod

        monkeypatch.setattr(cfgmod, "_CONFIG_FILE", tmp_path / "missing.toml")
        monkeypatch.delenv("AI_HOUKAI_SUMMARIZER", raising=False)
        assert cfgmod.load_maintenance().summarizer is None

    def test_scheduler_forwards_summarizer_to_reflection(self, monkeypatch, tmp_path):
        from ai_houkai.maintenance import scheduler as sched_mod
        from ai_houkai.maintenance.scheduler import MaintenanceScheduler

        seen = {}

        class FakeEngine:
            def __init__(self, store, **kw):
                seen.update(kw)

            def reflect(self, dry_run=False):
                return []

        monkeypatch.setattr(sched_mod, "ReflectionEngine", FakeEngine)

        marker = lambda mems: "x"  # noqa: E731
        sched = MaintenanceScheduler(
            store=object(),
            decay_every=None,
            reflect_every=1,
            state_path=str(tmp_path / "state.json"),
            summarizer=marker,
        )
        sched.tick(now=1_000_000.0)
        assert seen["summarizer"] is marker
