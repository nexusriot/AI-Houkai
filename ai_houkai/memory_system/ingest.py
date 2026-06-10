"""Chunking for bulk ingestion (`houkai ingest`).

``chunk_text()`` splits free-form text (notes, markdown, transcripts)
into memory-sized chunks:

1. Split on blank lines into paragraphs.
2. A markdown heading is glued to the paragraph that follows it, so the
   stored memory keeps its context ("## Deploy\\nUse make release").
3. Paragraphs longer than ``max_chars`` are re-split on sentence
   boundaries and greedily re-packed up to ``max_chars``.
4. Chunks shorter than ``min_chars`` are dropped (separators, stray
   list bullets, noise).

Deterministic and dependency-free; embedding and storage happen at the
caller (one ``remember()`` per chunk).
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^#{1,6}\s+\S")


def _split_long(paragraph: str, max_chars: int) -> list[str]:
    """Greedily pack sentences into chunks of at most max_chars."""
    sentences = _SENTENCE_SPLIT.split(paragraph)
    chunks: list[str] = []
    current = ""
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        candidate = f"{current} {sent}".strip() if current else sent
        if len(candidate) <= max_chars or not current:
            current = candidate
        else:
            chunks.append(current)
            current = sent
    if current:
        chunks.append(current)
    # A single sentence longer than max_chars is kept whole — splitting
    # mid-sentence would destroy the embedding's meaning.
    return chunks


def chunk_text(
    text: str,
    *,
    max_chars: int = 500,
    min_chars: int = 30,
) -> list[str]:
    """Split *text* into memory-sized chunks (see module docstring)."""
    normalized = text.replace("\r\n", "\n")
    blocks = [b.strip() for b in re.split(r"\n\s*\n", normalized)]
    blocks = [b for b in blocks if b]

    # Glue headings onto the following block.
    merged: list[str] = []
    pending_heading: str | None = None
    for block in blocks:
        if _HEADING.match(block) and "\n" not in block:
            pending_heading = (
                f"{pending_heading}\n{block}" if pending_heading else block
            )
            continue
        if pending_heading:
            block = f"{pending_heading}\n{block}"
            pending_heading = None
        merged.append(block)
    if pending_heading:
        merged.append(pending_heading)

    chunks: list[str] = []
    for block in merged:
        if len(block) <= max_chars:
            chunks.append(block)
        else:
            chunks.extend(_split_long(block, max_chars))

    return [c for c in chunks if len(c) >= min_chars]
