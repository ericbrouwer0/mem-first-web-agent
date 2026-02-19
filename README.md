# Memory-First Web Agent

A memory-first web agent that answers questions by checking a Redis vector cache before falling back to live web search. Built with LangGraph, OpenAI, and Tavily.

## Quick Start

**Prerequisites:** Docker, Python 3.13+, [uv](https://docs.astral.sh/uv/) (or pip)

```bash
# 1. Start Redis Stack (includes vector search)
docker compose up -d

# 2. Configure environment
cp .env.example .env
# Fill in OPENAI_API_KEY and TAVILY_API_KEY

# 3. Install dependencies
uv sync

# 4. Run
python main.py "What is LangGraph?"        # single query
python main.py -i                           # interactive REPL
```

## How It Works

```
User Query
    |
    v
Embed Query (OpenAI)
    |
    v
Vector Search (Redis) -----> similarity >= 0.7? -----> YES --> Sufficiency Check
    |                                                              |         |
    | NO                                                     sufficient  insufficient
    |                                                           |            |
    v                                                           v            |
Web Search (Tavily) <--------------------------------------------+           |
    |                                                                        |
    v                                                                        |
Chunk + Embed + Store in Redis                                               |
    |                                                                        |
    +------------------------------------------------------------------------+
    |
    v
Generate Response (LLM)
    |
    v
Log Interaction --> Return Answer + Sources
```

The agent first checks Redis for cached knowledge. If a match is found, a sufficiency check verifies the context actually answers the question -- if not, it falls back to web search automatically. All web results are chunked, embedded, and stored in Redis for future queries.

## Project Structure

```
src/
  agent/          State machine (LangGraph)
    state.py        AgentState TypedDict -- data flowing between nodes
    nodes.py        Node functions (embed, search, route, generate, log)
    graph.py        Graph wiring with conditional edges

  memory/         Redis vector store
    embedder.py     OpenAI embeddings (text-embedding-3-small, 1536-dim)
    redis_client.py RediSearch FLAT index, CRUD, KNN vector search
    vector_search.py High-level facade combining embedder + Redis

  web/            Web search and content extraction
    search_provider.py  Abstract SearchProvider interface
    tavily_search.py    Tavily API implementation
    fetcher.py          httpx page fetcher (fallback)
    converter.py        HTML-to-Markdown via markdownify

  llm/            Model-agnostic LLM wrappers
    base.py         BaseLLM -- works with any OpenAI-compatible API
    conversation.py Response generation (default: gpt-5.2)
    analytics.py    Query classification + sufficiency checks (default: gpt-5-mini)

  security/       Input validation
    guardrails.py   Prompt injection detection + query length validation

  utils/          Shared utilities
    config.py       Settings dataclass (reads from .env)
    logger.py       Structured logging + JSONL analytics
    chunker.py      Overlapping text chunker (500 tokens, 50 overlap)
```

## Configuration

All settings are read from environment variables. See `.env.example` for the full list. Key settings:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `TAVILY_API_KEY` | (required) | Tavily search API key |
| `OPENAI_CONVERSATION_MODEL` | `gpt-5.2` | Model for response generation |
| `OPENAI_ANALYTICS_MODEL` | `gpt-5-mini` | Model for classification/sufficiency |
| `SIMILARITY_THRESHOLD` | `0.7` | Cosine similarity cutoff for memory hits |

Models are injected via config -- swap to any OpenAI-compatible model by changing the env var.

## Tests

```bash
# Unit tests (no external services needed for mocked tests)
uv run pytest tests/unit -v

# Integration tests (requires running Redis + valid API keys)
uv run pytest tests/integration -v
```

## Infrastructure

- **Redis Stack** (`docker-compose.yml`) -- `redis/redis-stack:latest` provides RediSearch for vector indexing. Ports: 6379 (Redis), 8001 (RedisInsight UI).
- **LangGraph** -- State machine framework. Each node receives the full state and returns a partial update.
- **Tavily** -- AI-optimized search API. Returns pre-extracted content, reducing the need for HTML parsing.
