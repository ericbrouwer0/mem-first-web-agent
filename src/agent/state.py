"""Agent state schema -- the single TypedDict that flows through every node."""

from typing import Any, Dict, List, Optional, TypedDict


class AgentState(TypedDict, total=False):
    """State carried across the LangGraph state machine.

    Every node receives the full state and returns a *partial* dict with only
    the keys it wants to update.
    """

    # Input
    query: str

    # Embedding
    query_embedding: Optional[List[float]]

    # Memory search
    memory_results: Optional[List[Dict[str, Any]]]
    best_similarity: Optional[float]
    memory_hit: bool

    # Web search / fetch
    search_results: Optional[list]
    fetched_content: Optional[Dict[str, str]]
    chunks: Optional[List[Dict[str, Any]]]

    # Response generation
    context: List[str]
    metadata: List[Dict[str, Any]]
    response: Optional[str]

    # Routing / observability
    route_taken: str          # "memory_hit" | "memory_miss"
    start_time: float
    end_time: Optional[float]
