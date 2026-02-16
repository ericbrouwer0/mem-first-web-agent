"""Content chunker -- splits text into overlapping chunks for embedding."""

import uuid
import time
from typing import Any, Dict, List

from src.utils.config import settings


def estimate_tokens(text: str) -> int:
    """Rough token estimate (1 token ~ 0.75 words)."""
    return max(1, int(len(text.split()) / 0.75))


def chunk_text(
    content: str,
    source_url: str = "",
    title: str = "",
    query_context: str = "",
) -> List[Dict[str, Any]]:
    """Split *content* into overlapping chunks and return chunk dicts.

    Each dict is ready to be passed to ``RedisClient.store_chunk`` (minus the
    embedding, which ``VectorSearch.store`` adds).
    """
    words = content.split()
    if not words:
        return []

    # Convert token targets to approximate word counts (1 token ~ 0.75 words)
    chunk_words = max(1, int(settings.chunk_size * 0.75))
    overlap_words = max(0, int(settings.chunk_overlap * 0.75))
    step = max(1, chunk_words - overlap_words)

    raw_chunks: List[str] = []
    for start in range(0, len(words), step):
        segment = " ".join(words[start : start + chunk_words])
        if segment.strip():
            raw_chunks.append(segment)

    now = time.time()
    total = len(raw_chunks)
    results: List[Dict[str, Any]] = []
    for idx, text in enumerate(raw_chunks):
        results.append(
            {
                "chunk_id": str(uuid.uuid4()),
                "content": text,
                "source_url": source_url,
                "title": title,
                "created_at": now,
                "chunk_index": idx,
                "total_chunks": total,
                "query_context": query_context,
            }
        )
    return results
