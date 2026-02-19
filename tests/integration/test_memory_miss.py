"""Integration test -- memory miss flow.

Requires: Redis Stack running, valid OPENAI_API_KEY and TAVILY_API_KEY in .env.
"""

import time
import uuid
import pytest

from src.agent.graph import build_graph
from src.memory.redis_client import RedisClient


@pytest.mark.integration
def test_full_memory_miss_flow():
    """Query with empty Redis -- agent should fall back to web search."""
    try:
        rc = RedisClient()
        rc.ping()
    except Exception:
        pytest.skip("Redis not available")

    rc.flush_index()

    graph = build_graph()
    unique_query = f"What is LangGraph {uuid.uuid4()}"  # ensures cache miss

    result = graph.invoke({
        "query": unique_query,
        "query_embedding": None,
        "memory_results": None,
        "best_similarity": None,
        "memory_hit": False,
        "search_results": None,
        "fetched_content": None,
        "chunks": None,
        "context": [],
        "metadata": [],
        "response": None,
        "route_taken": "",
        "start_time": time.time(),
        "end_time": None,
    })

    assert result["memory_hit"] is False
    assert result["route_taken"] == "memory_miss"
    assert result["response"] is not None
    assert result["chunks"] is not None
    assert len(result["chunks"]) > 0

    rc.flush_index()
