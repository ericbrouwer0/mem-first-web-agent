"""Integration test -- memory hit flow.

Requires: Redis Stack running, valid OPENAI_API_KEY in .env.
"""

import time
import pytest

from src.agent.graph import build_graph
from src.memory.embedder import Embedder
from src.memory.redis_client import RedisClient


@pytest.mark.integration
def test_full_memory_hit_flow():
    """Pre-populate Redis, query the agent, verify the memory-hit path fires."""
    try:
        rc = RedisClient()
        rc.ping()
    except Exception:
        pytest.skip("Redis not available")

    rc.flush_index()

    embedder = Embedder()
    content = (
        "LangGraph is a library for building stateful, multi-actor applications "
        "with LLMs. It extends LangChain Expression Language with the ability to "
        "coordinate multiple chains across multiple steps of computation."
    )
    vec = embedder.embed_text(content)
    rc.store_chunk("lg-intro", {
        "chunk_id": "lg-intro",
        "content": content,
        "embedding": vec,
        "source_url": "https://langchain.com/langgraph",
        "title": "LangGraph Introduction",
        "created_at": time.time(),
        "chunk_index": 0,
        "total_chunks": 1,
        "query_context": "langgraph information",
    })

    graph = build_graph()
    result = graph.invoke({
        "query": "What is LangGraph?",
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

    assert result["memory_hit"] is True
    assert result["route_taken"] == "memory_hit"
    assert result["best_similarity"] >= 0.7
    assert result["response"] is not None
    assert len(result["response"]) > 0

    rc.flush_index()
