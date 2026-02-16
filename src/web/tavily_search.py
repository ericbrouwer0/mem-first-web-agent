"""Tavily Search implementation (our sole web search provider)."""

from typing import List

from tavily import TavilyClient

from src.utils.config import settings
from src.utils.logger import get_logger
from src.web.search_provider import SearchProvider, SearchResult

log = get_logger(__name__)


class TavilySearch(SearchProvider):
    """Web search via the Tavily API.

    Tavily returns pre-extracted, LLM-ready content alongside each result,
    which means we can often skip the fetch-and-convert step entirely.
    """

    def __init__(self, api_key: str | None = None):
        self._client = TavilyClient(api_key=api_key or settings.tavily_api_key)

    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Execute a Tavily search and normalise results."""
        try:
            raw = self._client.search(
                query=query,
                max_results=num_results,
                include_raw_content=False,
            )
        except Exception:
            log.exception("Tavily search failed for query: %s", query)
            return []

        results: List[SearchResult] = []
        for item in raw.get("results", []):
            results.append(
                SearchResult(
                    url=item.get("url", ""),
                    title=item.get("title", ""),
                    snippet=item.get("content", ""),
                    content=item.get("content", ""),
                    relevance_score=item.get("score"),
                )
            )
        return results
