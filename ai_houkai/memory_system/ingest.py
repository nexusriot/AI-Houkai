"""Chunking for bulk ingestion (`houkai ingest`).

``chunk_text()`` splits free-form text (notes, markdown, transcripts)
into memory-sized chunks:

1. Split on blank lines into paragraphs.
2. A markdown heading is glued to the paragraph that follows it, so the
   stored memory keeps its context ("## Deploy\\nUse make release").
3. Paragraphs longer than ``max_chars`` are re-split on sentence
   boundaries and greedily re-packed up to ``max_chars``.
4. Blocks shorter than ``min_chars`` are dropped (separators, stray
   list bullets, noise). A short *fragment* of a split paragraph is folded
   into a neighbour instead, because it is content rather than noise.

Deterministic and dependency-free; embedding and storage happen at the
caller (one ``remember()`` per chunk).
"""

from __future__ import annotations

import re

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^#{1,6}\s+\S")


def _split_long(paragraph: str, max_chars: int, min_chars: int) -> list[str]:
    """Greedily pack sentences into chunks of at most max_chars.

    Fragments below *min_chars* are folded into a neighbour rather than left for
    the caller's noise filter to delete. Greedy packing leaves a short fragment
    whenever the next sentence will not fit, and that fragment is the tail (or
    head) of a real paragraph — content, not a separator. ``max_chars`` is
    already a soft target, since a single over-long sentence is kept whole, so
    growing a chunk is the lesser evil against silently losing text.
    """
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
    if min_chars <= 0 or len(chunks) < 2:
        return chunks

    folded: list[str] = []
    for chunk in chunks:
        if folded and len(chunk) < min_chars:
            folded[-1] = f"{folded[-1]} {chunk}"
        else:
            folded.append(chunk)
    # A runt in first position has no predecessor to join, so fold it forward.
    if len(folded) > 1 and len(folded[0]) < min_chars:
        folded[1] = f"{folded[0]} {folded[1]}"
        folded.pop(0)
    return folded


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
            chunks.extend(_split_long(block, max_chars, min_chars))

    return [c for c in chunks if len(c) >= min_chars]
