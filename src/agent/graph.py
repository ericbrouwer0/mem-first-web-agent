"""LangGraph state machine wiring.

State flows:

  embed_query -> vector_search -> [route_decision]
                                      |
                            +---------+---------+
                            |                   |
                        (memory)              (web)
                            |                   |
                  prepare_memory_ctx     web_search
                            |                   |
                            |            store_web_results
                            |                   |
                            +----> generate_response
                                        |
                                  log_interaction
                                        |
                                       END
"""

from langgraph.graph import END, StateGraph

from src.agent.nodes import (
    embed_query_node,
    generate_response_node,
    log_interaction_node,
    prepare_memory_context_node,
    route_decision_node,
    store_web_results_node,
    vector_search_node,
    web_search_node,
)
from src.agent.state import AgentState


def build_graph() -> StateGraph:
    """Construct and compile the agent graph.  Returns a runnable."""
    g = StateGraph(AgentState)

    # -- add nodes ----------------------------------------------------------
    g.add_node("embed_query", embed_query_node)
    g.add_node("vector_search", vector_search_node)
    g.add_node("prepare_memory_ctx", prepare_memory_context_node)
    g.add_node("web_search", web_search_node)
    g.add_node("store_web_results", store_web_results_node)
    g.add_node("generate_response", generate_response_node)
    g.add_node("log_interaction", log_interaction_node)

    # -- edges --------------------------------------------------------------
    g.set_entry_point("embed_query")
    g.add_edge("embed_query", "vector_search")

    # Conditional: memory hit vs miss
    g.add_conditional_edges(
        "vector_search",
        route_decision_node,
        {
            "memory": "prepare_memory_ctx",
            "web": "web_search",
        },
    )

    # Memory-hit path
    g.add_edge("prepare_memory_ctx", "generate_response")

    # Memory-miss path
    g.add_edge("web_search", "store_web_results")
    g.add_edge("store_web_results", "generate_response")

    # Common tail
    g.add_edge("generate_response", "log_interaction")
    g.add_edge("log_interaction", END)

    return g.compile()
