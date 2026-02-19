"""Redis connection, vector index management, and CRUD operations."""

from typing import Any, Dict, List, Optional

import numpy as np
import redis
from redis.commands.search.field import (
    NumericField,
    TextField,
    VectorField,
)
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from src.memory.embedder import EMBEDDING_DIM
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

INDEX_NAME = "memory_chunks"
KEY_PREFIX = "chunk:"


class RedisClient:
    """Thin wrapper around redis-py that manages vector index + chunk CRUD."""

    def __init__(
        self,
        host: str | None = None,
        port: int | None = None,
        password: str | None = None,
    ):
        self.client = redis.Redis(
            host=host or settings.redis_host,
            port=port or settings.redis_port,
            password=password or settings.redis_password or None,
            decode_responses=True,
        )
        self._ensure_index()

    # -- Index management ---------------------------------------------------

    def _ensure_index(self) -> None:
        """Create the RediSearch vector index if it does not already exist."""
        try:
            self.client.ft(INDEX_NAME).info()
            log.info("Redis index '%s' already exists", INDEX_NAME)
        except redis.ResponseError:
            schema = [
                TextField("content"),
                TextField("source_url"),
                TextField("title"),
                NumericField("created_at"),
                NumericField("chunk_index"),
                NumericField("total_chunks"),
                TextField("query_context"),
                VectorField(
                    "embedding",
                    "FLAT",
                    {
                        "TYPE": "FLOAT32",
                        "DIM": EMBEDDING_DIM,
                        "DISTANCE_METRIC": "COSINE",
                    },
                ),
            ]
            self.client.ft(INDEX_NAME).create_index(
                schema,
                definition=IndexDefinition(prefix=[KEY_PREFIX], index_type=IndexType.HASH),
            )
            log.info("Created Redis index '%s'", INDEX_NAME)

    # -- Chunk storage ------------------------------------------------------

    def store_chunk(self, chunk_id: str, data: Dict[str, Any]) -> bool:
        """Store a single chunk (embedding must be a list of floats)."""
        mapping = {k: v for k, v in data.items() if k != "embedding"}
        mapping["embedding"] = _vector_to_bytes(data["embedding"])
        self.client.hset(f"{KEY_PREFIX}{chunk_id}", mapping=mapping)
        return True

    def batch_store_chunks(self, chunks: List[Dict[str, Any]]) -> int:
        """Store multiple chunks in a pipeline. Returns count stored."""
        pipe = self.client.pipeline()
        for chunk in chunks:
            cid = chunk["chunk_id"]
            mapping = {k: v for k, v in chunk.items() if k != "embedding"}
            mapping["embedding"] = _vector_to_bytes(chunk["embedding"])
            pipe.hset(f"{KEY_PREFIX}{cid}", mapping=mapping)
        pipe.execute()
        return len(chunks)

    # -- Vector search ------------------------------------------------------

    def vector_search(
        self,
        query_vector: List[float],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return top_k chunks sorted by cosine similarity (highest first)."""
        q = (
            Query(f"*=>[KNN {top_k} @embedding $vec AS score]")
            .sort_by("score")
            .return_fields("content", "source_url", "title", "score", "query_context")
            .dialect(2)
        )
        raw = self.client.ft(INDEX_NAME).search(
            q, query_params={"vec": _vector_to_bytes(query_vector)}
        )
        results: List[Dict[str, Any]] = []
        for doc in raw.docs:
            cosine_dist = float(doc.score)
            similarity = 1.0 - cosine_dist
            results.append(
                {
                    "chunk_id": doc.id.replace(KEY_PREFIX, ""),
                    "content": doc.content,
                    "source_url": getattr(doc, "source_url", ""),
                    "title": getattr(doc, "title", ""),
                    "similarity": round(similarity, 4),
                }
            )
        return results

    # -- Utilities ----------------------------------------------------------

    def flush_index(self) -> None:
        """Drop and recreate the index (useful in tests)."""
        try:
            self.client.ft(INDEX_NAME).dropindex(delete_documents=True)
        except redis.ResponseError:
            pass
        self._ensure_index()

    def ping(self) -> bool:
        return self.client.ping()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _vector_to_bytes(vector: List[float]) -> bytes:
    """Convert a Python list of floats to the bytes blob Redis expects."""
    return np.array(vector, dtype=np.float32).tobytes()
