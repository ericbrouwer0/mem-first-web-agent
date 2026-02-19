"""Node implementations for the LangGraph agent.

Each function receives the full ``AgentState`` and returns a *partial* dict
with only the keys it updates.
"""

import time
from typing import Any, Dict

from src.agent.state import AgentState
from src.llm.conversation import ConversationLLM
from src.llm.analytics import AnalyticsLLM
from src.memory.embedder import Embedder
from src.memory.redis_client import RedisClient
from src.utils.chunker import chunk_text
from src.utils.config import settings
from src.utils.logger import get_logger, log_interaction
from src.web.tavily_search import TavilySearch

log = get_logger(__name__)

# Shared singletons (created on first use inside each node so import-time
# side effects are avoided).
_embedder: Embedder | None = None
_redis: RedisClient | None = None
_searcher: TavilySearch | None = None
_conv_llm: ConversationLLM | None = None
_analytics_llm: AnalyticsLLM | None = None


def _get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder


def _get_redis() -> RedisClient:
    global _redis
    if _redis is None:
        _redis = RedisClient()
    return _redis


def _get_searcher() -> TavilySearch:
    global _searcher
    if _searcher is None:
        _searcher = TavilySearch()
    return _searcher


def _get_conv_llm() -> ConversationLLM:
    global _conv_llm
    if _conv_llm is None:
        _conv_llm = ConversationLLM()
    return _conv_llm


def _get_analytics_llm() -> AnalyticsLLM:
    global _analytics_llm
    if _analytics_llm is None:
        _analytics_llm = AnalyticsLLM()
    return _analytics_llm


# ---- Nodes ---------------------------------------------------------------


def embed_query_node(state: AgentState) -> Dict[str, Any]:
    """Embed the user query and stamp the start time."""
    log.info("Embedding query: %s", state["query"])
    vector = _get_embedder().embed_text(state["query"])
    return {
        "query_embedding": vector,
        "start_time": time.time(),
    }


def vector_search_node(state: AgentState) -> Dict[str, Any]:
    """Search Redis for similar chunks and decide memory_hit / miss."""
    results = _get_redis().vector_search(
        state["query_embedding"], top_k=3
    )
    best = results[0]["similarity"] if results else 0.0
    hit = best >= settings.similarity_threshold
    log.info("Vector search: best_similarity=%.4f, memory_hit=%s", best, hit)
    return {
        "memory_results": results,
        "best_similarity": best,
        "memory_hit": hit,
    }


def route_decision_node(state: AgentState) -> str:
    """Conditional edge: return 'memory' or 'web' based on ``memory_hit``."""
    return "memory" if state.get("memory_hit") else "web"


def prepare_memory_context_node(state: AgentState) -> Dict[str, Any]:
    """Build context + metadata from memory results."""
    results = state.get("memory_results") or []
    context = [r["content"] for r in results]
    metadata = [
        {"source_url": r.get("source_url", ""), "title": r.get("title", "")}
        for r in results
    ]
    return {
        "context": context,
        "metadata": metadata,
        "route_taken": "memory_hit",
    }


def web_search_node(state: AgentState) -> Dict[str, Any]:
    """Run a Tavily search and store the results on state."""
    log.info("Memory miss -- falling back to Tavily search")
    results = _get_searcher().search(
        state["query"], num_results=settings.max_search_results
    )
    return {"search_results": results}


def store_web_results_node(state: AgentState) -> Dict[str, Any]:
    """Chunk, embed, and store Tavily results in Redis."""
    search_results = state.get("search_results") or []
    all_chunks = []
    for sr in search_results[: settings.max_pages_to_fetch]:
        chunks = chunk_text(
            content=sr.content,
            source_url=sr.url,
            title=sr.title,
            query_context=state["query"],
        )
        all_chunks.extend(chunks)

    if all_chunks:
        texts = [c["content"] for c in all_chunks]
        vectors = _get_embedder().embed_batch(texts)
        for chunk, vec in zip(all_chunks, vectors):
            chunk["embedding"] = vec
        _get_redis().batch_store_chunks(all_chunks)
        log.info("Stored %d chunks from web results", len(all_chunks))

    context = [c["content"] for c in all_chunks]
    metadata = [
        {"source_url": c["source_url"], "title": c["title"]}
        for c in all_chunks
    ]
    return {
        "chunks": all_chunks,
        "context": context,
        "metadata": metadata,
        "route_taken": "memory_miss",
    }


def generate_response_node(state: AgentState) -> Dict[str, Any]:
    """Call the conversation LLM to produce the final answer."""
    log.info("Generating response via %s", settings.conversation_model)
    response = _get_conv_llm().generate_response(
        query=state["query"],
        context=state.get("context", []),
        metadata=state.get("metadata", []),
    )
    return {"response": response or "(No response generated.)"}


def log_interaction_node(state: AgentState) -> Dict[str, Any]:
    """Log the interaction and stamp end_time."""
    end = time.time()
    elapsed_ms = (end - state.get("start_time", end)) * 1000
    sources = [m.get("source_url", "") for m in state.get("metadata", [])]

    # Fire-and-forget analytics (best effort)
    try:
        analytics = _get_analytics_llm().analyze_query(state["query"])
    except Exception:
        analytics = {}

    log_interaction(
        query=state["query"],
        route=state.get("route_taken", "unknown"),
        similarity_score=state.get("best_similarity", 0.0),
        sources=sources,
        response_time_ms=elapsed_ms,
        analytics=analytics,
    )
    log.info("Done -- route=%s, time=%.0fms", state.get("route_taken"), elapsed_ms)
    return {"end_time": end}
