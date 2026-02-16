"""CLI entry point for the memory-first web agent."""

import argparse
import sys
import time

from src.agent.graph import build_graph
from src.agent.state import AgentState
from src.security.guardrails import validate_query
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)


def run_query(query: str, graph) -> None:
    """Run a single query through the agent and print results."""
    # Validate
    ok, reason = validate_query(query)
    if not ok:
        print(f"\nError: {reason}\n")
        return

    print(f"\nQuery: {query}")
    print("Processing...\n")

    initial_state: AgentState = {
        "query": query,
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
    }

    result = graph.invoke(initial_state)

    route = result.get("route_taken", "unknown")
    similarity = result.get("best_similarity", 0.0)
    elapsed_ms = (result.get("end_time", time.time()) - initial_state["start_time"]) * 1000

    print(f"Route: {'Memory Hit' if route == 'memory_hit' else 'Memory Miss'} "
          f"(similarity: {similarity:.2f})")
    print(f"\nResponse:\n{result.get('response', '(no response)')}\n")

    sources = result.get("metadata", [])
    if sources:
        print("Sources:")
        seen = set()
        for i, m in enumerate(sources, 1):
            url = m.get("source_url", "")
            if url and url not in seen:
                print(f"  [{i}] {url}")
                seen.add(url)

    print(f"\nPerformance: {elapsed_ms:.0f}ms\n")


def interactive_mode(graph) -> None:
    """REPL loop for interactive querying."""
    print("Memory-First Web Agent  (type 'quit' or 'exit' to stop)\n")
    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break
        if not query:
            continue
        run_query(query, graph)


def main() -> None:
    parser = argparse.ArgumentParser(description="Memory-first web agent")
    parser.add_argument("query", nargs="?", help="Single query to run")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="Start interactive REPL mode")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable DEBUG logging")
    args = parser.parse_args()

    if args.verbose:
        import logging
        logging.getLogger().setLevel(logging.DEBUG)

    graph = build_graph()

    if args.interactive:
        interactive_mode(graph)
    elif args.query:
        run_query(args.query, graph)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
