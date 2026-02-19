"""Configuration management -- reads from environment with sensible defaults."""

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Centralised settings read once from env vars."""

    # --- OpenAI -----------------------------------------------------------
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    conversation_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_CONVERSATION_MODEL", "gpt-5.2")
    )
    analytics_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_ANALYTICS_MODEL", "gpt-5-mini")
    )

    # --- Redis -------------------------------------------------------------
    redis_host: str = field(default_factory=lambda: os.getenv("REDIS_HOST", "localhost"))
    redis_port: int = field(default_factory=lambda: int(os.getenv("REDIS_PORT", "6379")))
    redis_password: str = field(default_factory=lambda: os.getenv("REDIS_PASSWORD", ""))

    # --- Tavily ------------------------------------------------------------
    tavily_api_key: str = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))

    # --- Agent behaviour ---------------------------------------------------
    similarity_threshold: float = field(
        default_factory=lambda: float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    )
    chunk_size: int = field(default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500")))
    chunk_overlap: int = field(default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50")))
    max_search_results: int = field(
        default_factory=lambda: int(os.getenv("MAX_SEARCH_RESULTS", "5"))
    )
    max_pages_to_fetch: int = field(
        default_factory=lambda: int(os.getenv("MAX_PAGES_TO_FETCH", "3"))
    )

    # --- Logging -----------------------------------------------------------
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))
    log_file: str = field(default_factory=lambda: os.getenv("LOG_FILE", "logs/agent.log"))
    analytics_file: str = field(
        default_factory=lambda: os.getenv("ANALYTICS_FILE", "logs/analytics.jsonl")
    )

    # --- Guardrails --------------------------------------------------------
    max_query_length: int = field(
        default_factory=lambda: int(os.getenv("MAX_QUERY_LENGTH", "500"))
    )


# Module-level singleton -- import this everywhere.
settings = Settings()
