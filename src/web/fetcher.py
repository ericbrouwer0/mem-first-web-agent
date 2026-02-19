"""Web page fetcher (fallback when Tavily content is insufficient)."""

from typing import Dict, Optional

import httpx

from src.utils.logger import get_logger

log = get_logger(__name__)


def fetch_page(url: str, timeout: float = 10.0) -> Optional[str]:
    """GET a URL and return the body text, or None on any error."""
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True) as client:
            resp = client.get(url)
            resp.raise_for_status()
            return resp.text
    except Exception:
        log.warning("Failed to fetch %s", url)
        return None


def fetch_pages(urls: list[str], timeout: float = 10.0) -> Dict[str, Optional[str]]:
    """Fetch multiple URLs and return a url -> html mapping."""
    results: Dict[str, Optional[str]] = {}
    for url in urls:
        results[url] = fetch_page(url, timeout=timeout)
    return results
