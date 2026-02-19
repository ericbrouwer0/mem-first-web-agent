"""Structured logging and per-interaction analytics logging."""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """Return a configured logger (creates handler only once per name)."""
    from src.utils.config import settings

    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    effective_level = getattr(logging, (level or settings.log_level).upper(), logging.INFO)
    logger.setLevel(effective_level)

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)

    # Optional file handler
    log_path = Path(settings.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(fmt))
    logger.addHandler(fh)

    return logger


def log_interaction(
    query: str,
    route: str,
    similarity_score: float,
    sources: List[str],
    response_time_ms: float,
    analytics: Optional[Dict[str, Any]] = None,
) -> None:
    """Append a single interaction record to the JSONL analytics file."""
    from src.utils.config import settings

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "query": query,
        "route": route,
        "similarity_score": round(similarity_score, 4),
        "sources_used": sources,
        "response_time_ms": round(response_time_ms, 1),
        "analytics": analytics or {},
    }

    path = Path(settings.analytics_file)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
