"""LLM-backed reflection summarizers, built from a config spec string.

A summarizer is any ``Callable[[list[Memory]], str]`` (see ReflectionEngine).
``build_summarizer()`` turns a spec string into one:

    "extractive"               built-in extractive default (no LLM)
    "ollama:llama3.1"          local Ollama, OpenAI-compat endpoint (stdlib
                               urllib — no SDK needed; OLLAMA_BASE_URL env)
    "openai:gpt-4o-mini"       OpenAI SDK (pip install "ai-houkai[openai]")
    "anthropic:claude-haiku-4-5"  Anthropic SDK (pip install "ai-houkai[claude]")

The model part is passed through verbatim, so Ollama tags like
``ollama:llama3.1:8b`` work.

By default the returned callable is wrapped with a fallback: if the LLM
call fails (network down, key missing at call time, empty response) the
built-in extractive summarizer is used instead and a warning is logged —
reflection should degrade, not crash, when run unattended by the
maintenance daemon.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import TYPE_CHECKING, Callable

from .reflection import _default_summarizer

if TYPE_CHECKING:
    from .store import Memory

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None  # type: ignore

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # type: ignore

log = logging.getLogger(__name__)

Summarizer = Callable[["list[Memory]"], str]

PROVIDERS = ("extractive", "ollama", "openai", "anthropic")

_DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
_MAX_TOKENS = 1024

PROMPT_TEMPLATE = (
    "You are condensing an AI agent's episodic memories into one durable "
    "semantic memory.\n\nEvents (most important first):\n{events}\n\n"
    "Write a single concise summary (1-3 sentences) that captures the "
    "durable facts, decisions, and patterns. Do not add information that "
    "is not in the events. Output only the summary text."
)


def render_prompt(memories: "list[Memory]") -> str:
    ordered = sorted(memories, key=lambda m: m.importance, reverse=True)
    events = "\n".join(f"- {m.text}" for m in ordered)
    return PROMPT_TEMPLATE.format(events=events)


def _ollama_summarizer(model: str) -> Summarizer:
    """Chat against Ollama's OpenAI-compatible endpoint using stdlib only."""

    def summarize(memories: "list[Memory]") -> str:
        base = os.environ.get("OLLAMA_BASE_URL", _DEFAULT_OLLAMA_BASE_URL)
        base = base.rstrip("/")
        if not base.endswith("/v1"):
            base += "/v1"
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": render_prompt(memories)}],
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{base}/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    return summarize


def _openai_summarizer(model: str) -> Summarizer:
    if OpenAI is None:
        raise ImportError(
            "openai SDK is required for an 'openai:' summarizer — "
            'pip install "ai-houkai[openai]"'
        )

    def summarize(memories: "list[Memory]") -> str:
        # Built at call time so a missing/invalid key surfaces inside the
        # _with_fallback guard (degrade to extractive) rather than crashing
        # build_summarizer() — e.g. in the unattended maintenance daemon.
        client = OpenAI()
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": render_prompt(memories)}],
        )
        return resp.choices[0].message.content or ""

    return summarize


def _anthropic_summarizer(model: str) -> Summarizer:
    if Anthropic is None:
        raise ImportError(
            "anthropic SDK is required for an 'anthropic:' summarizer — "
            'pip install "ai-houkai[claude]"'
        )

    def summarize(memories: "list[Memory]") -> str:
        # Built at call time — see _openai_summarizer for the rationale.
        client = Anthropic()
        resp = client.messages.create(
            model=model,
            max_tokens=_MAX_TOKENS,
            messages=[{"role": "user", "content": render_prompt(memories)}],
        )
        return "".join(
            block.text for block in resp.content if getattr(block, "type", "") == "text"
        )

    return summarize


def _with_fallback(inner: Summarizer, spec: str) -> Summarizer:
    def summarize(memories: "list[Memory]") -> str:
        try:
            text = (inner(memories) or "").strip()
        except Exception as exc:
            log.warning(
                "Summarizer %r failed (%s) — falling back to extractive.",
                spec, exc,
            )
            return _default_summarizer(memories)
        if not text:
            log.warning(
                "Summarizer %r returned empty output — falling back to extractive.",
                spec,
            )
            return _default_summarizer(memories)
        return text

    return summarize


def build_summarizer(spec: str | None, *, fallback: bool = True) -> Summarizer:
    """Build a summarizer callable from a ``provider:model`` spec string.

    spec
        ``None``, ``""`` or ``"extractive"`` → built-in extractive default.
        Otherwise ``provider:model`` with provider one of
        ``ollama`` / ``openai`` / ``anthropic``.
    fallback
        Wrap LLM providers so failures degrade to the extractive
        summarizer with a logged warning instead of raising.

    Raises ValueError on an unknown provider or a missing model, and
    ImportError if the provider's SDK is not installed.
    """
    if not spec or spec in ("extractive", "default", "none"):
        return _default_summarizer

    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip()

    if provider == "extractive":
        return _default_summarizer
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown summarizer provider {provider!r} in spec {spec!r} — "
            f"expected one of {', '.join(PROVIDERS)}"
        )
    if not model:
        raise ValueError(
            f"Summarizer spec {spec!r} is missing a model — "
            f"expected e.g. '{provider}:MODEL'"
        )

    if provider == "ollama":
        inner = _ollama_summarizer(model)
    elif provider == "openai":
        inner = _openai_summarizer(model)
    else:
        inner = _anthropic_summarizer(model)

    return _with_fallback(inner, spec) if fallback else inner
