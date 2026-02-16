"""Web module -- Tavily search, page fetching, HTML conversion."""

from src.web.search_provider import SearchProvider, SearchResult
from src.web.tavily_search import TavilySearch

__all__ = ["SearchProvider", "SearchResult", "TavilySearch"]
