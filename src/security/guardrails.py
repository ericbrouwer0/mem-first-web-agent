"""Prompt injection detection and input validation."""

import re
from typing import Optional, Tuple

from src.utils.config import settings

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions",
    r"disregard\s+(your\s+)?(instructions|prompt)",
    r"you\s+are\s+(now|a)\s+",
    r"pretend\s+you\s+are",
    r"act\s+as\s+if\s+you\s+(are|were)",
    r"system\s*:\s*",
    r"<\s*script",
    r"';\s*DROP\s+TABLE",
    r"\[system\]",
    r"<\|.*\|>",
]


def check_injection(text: str) -> Tuple[bool, Optional[str]]:
    """Return (is_safe, reason).  ``is_safe`` is False when injection is suspected."""
    if not text or not text.strip():
        return True, None
    lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lower, re.IGNORECASE):
            return False, f"Potential injection detected: {pattern}"
    return True, None


def validate_query(query: str) -> Tuple[bool, Optional[str]]:
    """Validate length and basic sanity of a user query."""
    if not query or not query.strip():
        return False, "Query is empty."
    if len(query) > settings.max_query_length:
        return False, f"Query exceeds max length ({settings.max_query_length} chars)."
    is_safe, reason = check_injection(query)
    if not is_safe:
        return False, reason
    return True, None
