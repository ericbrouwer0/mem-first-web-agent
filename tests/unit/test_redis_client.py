"""Unit tests for the RedisClient.

These tests require a running redis-stack instance (docker-compose up -d).
Mark them with ``@pytest.mark.skipif`` if CI has no Redis.
"""

import time
import pytest
from unittest.mock import patch, MagicMock

from src.memory.redis_client import RedisClient, _vector_to_bytes


def test_vector_to_bytes_roundtrip():
    """Bytes conversion preserves values."""
    import numpy as np
    vec = [0.1, 0.2, 0.3]
    raw = _vector_to_bytes(vec)
    assert isinstance(raw, bytes)
    back = np.frombuffer(raw, dtype=np.float32).tolist()
    assert len(back) == 3
    assert abs(back[0] - 0.1) < 1e-6


@pytest.mark.integration
class TestRedisClientLive:
    """Tests that talk to a real Redis Stack instance."""

    @pytest.fixture(autouse=True)
    def setup(self):
        try:
            self.client = RedisClient()
            self.client.ping()
        except Exception:
            pytest.skip("Redis not available")
        self.client.flush_index()
        yield
        self.client.flush_index()

    def test_store_and_search(self):
        vec = [0.0] * 1536
        vec[0] = 1.0
        chunk = {
            "chunk_id": "test-1",
            "content": "LangGraph is great",
            "embedding": vec,
            "source_url": "https://example.com",
            "title": "Test",
            "created_at": time.time(),
            "chunk_index": 0,
            "total_chunks": 1,
            "query_context": "test",
        }
        self.client.store_chunk("test-1", chunk)

        results = self.client.vector_search(vec, top_k=1)
        assert len(results) == 1
        assert "LangGraph" in results[0]["content"]
        assert results[0]["similarity"] > 0.9
