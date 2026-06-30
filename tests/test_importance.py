"""Tests for heuristic importance auto-assignment."""

from __future__ import annotations

import pytest

from ai_houkai.cli import config as cfgmod
from ai_houkai.memory_system import MemoryStore
from ai_houkai.memory_system.importance import score_importance


class TestTiers:
    def test_standing_instruction_scores_high(self):
        assert score_importance("Always run make lint before committing") >= 0.9
        assert score_importance("Never push directly to main") >= 0.9
        assert score_importance("From now on use uv instead of pip") >= 0.9

    def test_correction_scores_high(self):
        assert score_importance("Actually, the API key lives in vault, not env") >= 0.9
        assert score_importance("That's wrong — the timeout is 30s") >= 0.9

    def test_preference_scores_high(self):
        assert score_importance("vlad prefers tabs over spaces") >= 0.9
        assert score_importance("I hate auto-formatting on save") >= 0.9

    def test_decision_scores_mid_high(self):
        s = score_importance("We decided to target Python 3.11 minimum")
        assert 0.7 <= s < 0.9
        s = score_importance("The team convention is one module per command")
        assert 0.7 <= s < 0.9

    def test_completion_scores_mid(self):
        s = score_importance("Fixed the race condition in the journal writer")
        assert 0.55 <= s < 0.7
        s = score_importance("Deployed version 0.5.5 to PyPI yesterday")
        assert 0.55 <= s < 0.7

    def test_observation_scores_low(self):
        assert score_importance("It seems the cache is sometimes stale") < 0.5
        assert score_importance("Noticed the tests take a while on CI") < 0.5

    def test_neutral_text_gets_default(self):
        assert score_importance("The store uses ChromaDB for persistence") == 0.5

    def test_strongest_tier_wins(self):
        # "never" (0.9) beats "maybe" (0.35)
        assert score_importance("Never use eval, maybe except in the REPL") >= 0.9


class TestModifiers:
    def test_procedural_type_bonus(self):
        base = score_importance("Deploy with make release", type="semantic")
        proc = score_importance("Deploy with make release", type="procedural")
        assert proc == pytest.approx(base + 0.10)

    def test_feedback_type_bonus(self):
        base = score_importance("The output was truncated", type="semantic")
        fb = score_importance("The output was truncated", type="feedback")
        assert fb == pytest.approx(base + 0.10)

    def test_question_penalty(self):
        plain = score_importance("The deploy target is staging")
        q = score_importance("Is the deploy target staging?")
        assert q < plain

    def test_short_fragment_penalty(self):
        assert score_importance("ok then") < 0.5

    def test_clamped_to_bounds(self):
        # instruction + procedural bonus must not exceed the ceiling
        s = score_importance("Always always never must", type="procedural")
        assert s <= 0.98
        # observation + question + short stays above the floor
        s = score_importance("hm, maybe?", type="semantic")
        assert s >= 0.05

    def test_deterministic(self):
        text = "We decided to use sqlite for the cache"
        assert score_importance(text) == score_importance(text)


class TestStoreWiring:
    def test_remember_without_importance_uses_fn(self, tmp_path):
        s = MemoryStore(
            path=str(tmp_path / "chroma"),
            collection="imp_test",
            importance_fn=score_importance,
        )
        try:
            mem = s.remember("Never commit secrets to the repo", type="procedural")
            assert mem.importance >= 0.9
            # explicit value still wins
            mem2 = s.remember("Never do that other thing", importance=0.2)
            assert mem2.importance == 0.2
        finally:
            s.client.close()

    def test_remember_without_fn_keeps_default(self, store):
        mem = store.remember("Never commit secrets to the repo")
        assert mem.importance == 0.5


class TestConfig:
    def test_default_importance_auto(self, monkeypatch, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text('default_importance = "auto"\n')
        monkeypatch.setattr(cfgmod, "_CONFIG_FILE", toml)
        assert cfgmod.load().default_importance == "auto"

    def test_default_importance_float_still_works(self, monkeypatch, tmp_path):
        toml = tmp_path / "config.toml"
        toml.write_text("default_importance = 0.7\n")
        monkeypatch.setattr(cfgmod, "_CONFIG_FILE", toml)
        assert cfgmod.load().default_importance == 0.7
