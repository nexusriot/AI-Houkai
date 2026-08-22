"""Pluggable embedding backends.

The store's default embedder is a local sentence-transformers model, which
pulls torch (~2 GB installed). That is the right default for a local-first
memory system, but it makes the library unusable in a slim container, on a
small SBC, or against a hosted embedding service — and it forces every test to
load a real model.

This module supplies stdlib-only alternatives that speak the OpenAI
``/v1/embeddings`` protocol, mirroring ``go/internal/embed``. They are plain
callables matching ChromaDB's embedding-function protocol
(``fn(list[str]) -> list[list[float]]``) and go straight into
``MemoryStore(embedding_function=...)``::

    from ai_houkai.embed import OpenAICompatibleEmbedder
    from ai_houkai.memory_system import MemoryStore

    store = MemoryStore(
        path="~/.ai_houkai/.chroma",
        embedding_function=OpenAICompatibleEmbedder(
            api_key=os.environ["OPENAI_API_KEY"],
            model="text-embedding-3-small",
        ),
    )

Or build one from a ``provider:model`` spec, the same grammar the reflection
summarizers use::

    build_embedder("ollama:nomic-embed-text")
    build_embedder("openai:text-embedding-3-small")
    build_embedder("digitalocean:gte-large-en-v1.5")
    build_embedder("local:all-MiniLM-L6-v2")      # sentence-transformers

**Changing the embedding model changes the vector space.** Vectors written by
one model are meaningless to another; ``houkai doctor`` carries an embed-dim
guardrail that catches the obvious case (different dimensions), but two models
of the same width will silently produce nonsense rankings. Re-embed via
export + ``import --regenerate-vectors`` when switching.
"""

from __future__ import annotations

import functools
import json
import logging
import os
import urllib.error
import urllib.request
from typing import Sequence

from chromadb.api.types import EmbeddingFunction
from chromadb.utils import embedding_functions

__all__ = [
    "EmbeddingError",
    "OpenAICompatibleEmbedder",
    "OllamaEmbedder",
    "PROVIDERS",
    "DIGITALOCEAN_BASE_URL",
    "as_chroma_embedding_function",
    "build_embedder",
    "local_embedder",
]

# Wire-compatible with OpenAI's /v1/embeddings.
DIGITALOCEAN_BASE_URL = "https://inference.do-ai.run"
_OPENAI_BASE_URL = "https://api.openai.com"
_OLLAMA_BASE_URL = "http://localhost:11434"

PROVIDERS = ("local", "openai", "ollama", "digitalocean")


class EmbeddingError(RuntimeError):
    """The embedding backend could not be reached or returned a bad payload."""


@functools.lru_cache(maxsize=None)
def local_embedder(model_name: str = "all-MiniLM-L6-v2"):
    """Return a cached local sentence-transformers embedding function.

    Loaded once per process per model name — subsequent calls return the same
    instance, avoiding redundant disk reads and RAM allocation.

    Side-effects applied before the first load:
      • HF_HUB_DISABLE_PROGRESS_BARS=1  — suppress "Loading weights" bars
      • huggingface_hub loggers → ERROR  — suppress unauthenticated-request
        warnings (the model lives in the local cache; no auth is needed)

    sentence-transformers is an optional dependency (it pulls torch, ~2 GB):
    installs that skip it must supply a different embedder instead.
    """
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    for _logger_name in (
        "huggingface_hub",
        "huggingface_hub.utils",
        "huggingface_hub.utils._http",
        "huggingface_hub.utils._headers",
    ):
        logging.getLogger(_logger_name).setLevel(logging.ERROR)

    try:
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=model_name
        )
    except ImportError as exc:
        raise ImportError(
            "The default local embedder needs sentence-transformers, which is "
            "not installed. Either `pip install \"ai-houkai[local]\"`, or pass "
            "an embedding_function= (see ai_houkai.embed for stdlib OpenAI / "
            "Ollama backends) or set AI_HOUKAI_EMBEDDER=provider:model."
        ) from exc


def _post_json(url: str, payload: dict, headers: dict, timeout: float) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **headers},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:500]
        raise EmbeddingError(f"{url} returned HTTP {exc.code}: {body}") from exc
    except urllib.error.URLError as exc:
        raise EmbeddingError(f"{url} unreachable: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise EmbeddingError(f"{url} returned invalid JSON: {exc}") from exc


def _unreconstructable(_input):
    raise EmbeddingError(
        "This collection was written with an in-process embedding_function, "
        "which cannot be rebuilt from the persisted config. Pass the same "
        "embedding_function= when opening the store."
    )


class _CallableAdapter(EmbeddingFunction):
    """Adapt a plain ``fn(list[str]) -> list[list[float]]`` to ChromaDB.

    Chroma's collection API needs more than a callable: the query path calls
    ``embed_query`` and the collection config reads ``name()``/``is_legacy``.
    Rather than force every caller to subclass Chroma's Protocol, the store
    wraps whatever it is handed — so ``embedding_function=lambda texts: …``
    just works.
    """

    def __init__(self, fn) -> None:
        self._fn = fn

    def __call__(self, input):  # noqa: A002 — Chroma's parameter name
        return self._fn(input)

    @staticmethod
    def name() -> str:
        # Chroma calls this on the class, not the instance, so it cannot carry
        # per-instance detail. Use `.wrapped` for the underlying callable.
        return "ai_houkai_callable"

    @property
    def wrapped(self):
        """The callable this adapter delegates to."""
        return self._fn

    def default_space(self):
        return "cosine"

    def get_config(self) -> dict:
        return {}

    @staticmethod
    def build_from_config(config: dict) -> "_CallableAdapter":
        """Rebuild from a persisted collection config.

        An in-process callable cannot be serialised, so the reconstructed
        adapter is a tombstone: it satisfies Chroma's config round-trip (which
        would otherwise warn on every open) and fails loudly only if something
        actually tries to embed through it. In practice nothing does — the
        store always passes its live embedding_function to
        get_or_create_collection, which takes precedence.
        """
        return _CallableAdapter(_unreconstructable)

    def is_legacy(self) -> bool:
        return False


def as_chroma_embedding_function(fn):
    """Return *fn* as something ChromaDB's collection API fully supports.

    Already-conforming embedding functions (including this module's) pass
    through untouched; a bare callable is wrapped in :class:`_CallableAdapter`.
    """
    if fn is None:
        return None
    if not callable(fn):
        raise TypeError(
            f"embedding_function must be callable, got {type(fn).__name__}")
    # Wrap unless it already answers everything Chroma's collection API asks
    # of an embedding function (the query path and the config serialiser).
    if all(hasattr(fn, attr) for attr in ("embed_query", "get_config", "is_legacy")):
        return fn
    return _CallableAdapter(fn)


class OpenAICompatibleEmbedder:
    """Embedder for any service speaking OpenAI's ``/v1/embeddings`` protocol.

    Covers OpenAI itself, DigitalOcean Serverless Inference, vLLM, llama.cpp's
    OpenAI-compat server, Together, and Ollama's ``/v1`` shim. Uses ``urllib``
    rather than the ``openai`` SDK so the dependency stays optional.

    ``batch_size`` bounds how many texts go in one request — providers cap the
    input array, and a bulk ``remember_many`` can otherwise exceed it.
    """

    def __init__(
        self,
        api_key: str = "",
        model: str = "text-embedding-3-small",
        base_url: str = _OPENAI_BASE_URL,
        *,
        timeout: float = 30.0,
        batch_size: int = 256,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = max(1, batch_size)

    def name(self) -> str:
        return f"ai_houkai.openai_compatible:{self.model}"


    def __call__(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002
        texts = list(input)
        if not texts:
            return []
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        url = f"{self.base_url}/v1/embeddings"
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            data = _post_json(url, {"model": self.model, "input": chunk},
                              headers, self.timeout)
            rows = data.get("data")
            if not isinstance(rows, list) or len(rows) != len(chunk):
                raise EmbeddingError(
                    f"{url} returned {len(rows) if isinstance(rows, list) else '?'} "
                    f"embeddings for {len(chunk)} inputs"
                )
            # The API documents index-ordered results, but does not guarantee
            # ordering — sort defensively so text[i] always maps to vector[i].
            rows = sorted(rows, key=lambda r: r.get("index", 0))
            out.extend([float(x) for x in r["embedding"]] for r in rows)
        return out


class OllamaEmbedder:
    """Embedder against Ollama's native ``/api/embed`` endpoint.

    Ollama also exposes an OpenAI-compatible ``/v1/embeddings``; this class
    targets the native route, which needs no API key and accepts a batch.
    """

    def __init__(
        self,
        model: str = "nomic-embed-text",
        base_url: str = _OLLAMA_BASE_URL,
        *,
        timeout: float = 60.0,
        batch_size: int = 64,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.batch_size = max(1, batch_size)

    def name(self) -> str:
        return f"ai_houkai.ollama:{self.model}"


    def __call__(self, input: Sequence[str]) -> list[list[float]]:  # noqa: A002
        texts = list(input)
        if not texts:
            return []
        url = f"{self.base_url}/api/embed"
        out: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            chunk = texts[start:start + self.batch_size]
            data = _post_json(url, {"model": self.model, "input": chunk}, {},
                              self.timeout)
            rows = data.get("embeddings")
            if not isinstance(rows, list) or len(rows) != len(chunk):
                raise EmbeddingError(
                    f"{url} returned {len(rows) if isinstance(rows, list) else '?'} "
                    f"embeddings for {len(chunk)} inputs"
                )
            out.extend([float(x) for x in row] for row in rows)
        return out


def build_embedder(spec: str | None):
    """Build an embedding function from a ``provider:model`` spec.

    ``None`` / empty returns ``None`` so callers fall through to the store's
    default local model. Providers: ``local`` (sentence-transformers),
    ``openai``, ``ollama``, ``digitalocean``.

    Credentials come from the environment, never the spec, so a spec string is
    safe to put in a config file or a process listing:
    ``AI_HOUKAI_EMBED_API_KEY`` (falling back to ``OPENAI_API_KEY`` /
    ``DIGITALOCEAN_API_KEY``) and ``AI_HOUKAI_EMBED_BASE_URL``.

    Raises ValueError on an unknown provider or a missing model, and
    ImportError if a local model is requested without sentence-transformers.
    """
    if not spec:
        return None
    provider, _, model = spec.partition(":")
    provider = provider.strip().lower()
    model = model.strip()
    if provider not in PROVIDERS:
        raise ValueError(
            f"Unknown embedder provider {provider!r} in spec {spec!r} — "
            f"expected one of {', '.join(PROVIDERS)}"
        )
    if not model:
        raise ValueError(f"Missing model in embedder spec {spec!r} — "
                         f"expected e.g. '{provider}:MODEL'")

    base_override = os.environ.get("AI_HOUKAI_EMBED_BASE_URL", "")

    if provider == "local":
        return local_embedder(model)
    if provider == "ollama":
        return OllamaEmbedder(model, base_override or _OLLAMA_BASE_URL)

    key_env = "OPENAI_API_KEY" if provider == "openai" else "DIGITALOCEAN_API_KEY"
    api_key = (os.environ.get("AI_HOUKAI_EMBED_API_KEY")
               or os.environ.get(key_env, ""))
    default_base = (_OPENAI_BASE_URL if provider == "openai"
                    else DIGITALOCEAN_BASE_URL)
    return OpenAICompatibleEmbedder(api_key, model, base_override or default_base)
