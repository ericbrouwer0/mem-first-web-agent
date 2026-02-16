"""High-level vector search facade that combines embedder + Redis client."""

from typing import Any, Dict, List

from src.memory.embedder import Embedder
from src.memory.redis_client import RedisClient
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


class VectorSearch:
    """Convenience layer: embed a query, search Redis, return ranked results."""

    def __init__(
        self,
        redis_client: RedisClient | None = None,
        embedder: Embedder | None = None,
    ):
        self.redis = redis_client or RedisClient()
        self.embedder = embedder or Embedder()

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Embed *query* and return the top_k nearest chunks from Redis."""
        vector = self.embedder.embed_text(query)
        return self.redis.vector_search(vector, top_k=top_k)

    def store(self, chunks: List[Dict[str, Any]]) -> int:
        """Embed and store a list of chunk dicts (must contain 'content')."""
        texts = [c["content"] for c in chunks]
        vectors = self.embedder.embed_batch(texts)
        for chunk, vec in zip(chunks, vectors):
            chunk["embedding"] = vec
        return self.redis.batch_store_chunks(chunks)
