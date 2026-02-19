"""Analytics LLM -- lightweight classification of user queries.

Default model: gpt-5-mini (configurable via OPENAI_ANALYTICS_MODEL).
"""

import json
from typing import Any, Dict, Optional

from src.llm.base import BaseLLM
from src.utils.config import settings
from src.utils.logger import get_logger

log = get_logger(__name__)

ANALYTICS_PROMPT = """Analyze this user query and extract structured information.

Query: {query}

Return JSON with:
- topic: main subject (1-3 words)
- intent: user's goal (learning, troubleshooting, comparison, etc.)
- category: domain (technology, health, finance, general, etc.)
- complexity: simple, medium, complex

Format: Valid JSON only, no explanation."""


class AnalyticsLLM(BaseLLM):
    """Cheap, fast model used for query classification and summaries."""

    def __init__(self, model: str | None = None, **kwargs):
        super().__init__(model=model or settings.analytics_model, **kwargs)

    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Classify *query* and return a structured dict."""
        user_content = ANALYTICS_PROMPT.format(query=query)
        messages = [{"role": "user", "content": user_content}]
        raw = self.complete(messages, temperature=0.0)
        if not raw:
            return _fallback(query)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            log.warning("Analytics model returned non-JSON: %s", raw[:200])
            return _fallback(query)


def _fallback(query: str) -> Dict[str, Any]:
    """Return a safe default when the analytics model fails."""
    return {
        "topic": "unknown",
        "intent": "unknown",
        "category": "general",
        "complexity": "medium",
    }
