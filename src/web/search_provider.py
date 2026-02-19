"""Abstract search interface and shared SearchResult dataclass."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class SearchResult:
    """A single web search result."""

    url: str
    title: str
    snippet: str
    content: str  # full extracted text (Tavily provides this)
    relevance_score: Optional[float] = None


class SearchProvider(ABC):
    """Abstract interface -- swap implementations without touching callers."""

    @abstractmethod
    def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        """Return a list of search results for *query*."""
        ...
