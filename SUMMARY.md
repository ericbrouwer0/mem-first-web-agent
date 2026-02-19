# Memory-First Web Agent -- Implementation Summary

**Date:** February 16, 2026
**Status:** Proof of Concept (PoC)

---

## What Was Built

A memory-first web agent that answers user questions by first checking a Redis
vector cache and, only when no sufficiently similar answer exists, falling back
to live web search via Tavily. The system is designed to be language-model
agnostic: model names are read from configuration, and every LLM call goes
through a single base wrapper that works with any OpenAI-compatible API.

---

## Component Breakdown

### 1. Configuration and Logging (`src/utils/`)

- **config.py** -- A frozen `Settings` dataclass that reads every tunable
  parameter from environment variables with sensible defaults. All modules
  import the singleton `settings` object rather than calling `os.getenv`
  directly.
- **logger.py** -- Provides a `get_logger` factory for structured console and
  file logging. A separate `log_interaction` function appends one JSON record
  per query to a JSONL analytics file for later inspection.
- **chunker.py** -- Splits arbitrary text into overlapping chunks (default 500
  tokens, 50-token overlap) and attaches metadata (source URL, title, chunk
  index). The output is a list of dicts ready for embedding and storage.

### 2. Memory Layer (`src/memory/`)

- **embedder.py** -- Synchronous wrapper around the OpenAI embeddings API
  (`text-embedding-3-small` by default). Supports single-text and batch
  embedding. Dimension is fixed at 1536.
- **redis_client.py** -- Manages the RediSearch vector index
  (`memory_chunks`). On first instantiation it creates a FLAT COSINE index if
  one does not already exist. Exposes `store_chunk`, `batch_store_chunks`, and
  `vector_search` (KNN top-k with cosine similarity).
- **vector_search.py** -- High-level facade that combines the embedder and
  Redis client into two calls: `search(query)` and `store(chunks)`.

### 3. Web Layer (`src/web/`)

- **search_provider.py** -- Abstract `SearchProvider` base class and a
  `SearchResult` dataclass. This interface allows swapping search backends
  without touching any other module.
- **tavily_search.py** -- The sole production implementation. Tavily returns
  pre-extracted, LLM-ready content alongside each result, which eliminates the
  need to fetch and parse pages in most cases.
- **fetcher.py** -- A simple `httpx`-based page fetcher kept as a fallback for
  any scenario where raw HTML is needed.
- **converter.py** -- Converts HTML to clean Markdown via `markdownify`.

### 4. LLM Layer (`src/llm/`)

- **base.py** -- `BaseLLM` is a thin, model-agnostic wrapper around the
  OpenAI chat completions API. Model name and API key are injected at
  construction time. Swapping to a different provider (or a local model
  behind an OpenAI-compatible proxy) requires only changing the config.
- **conversation.py** -- `ConversationLLM` extends `BaseLLM` (default model:
  `gpt-5.2`). Its `generate_response` method builds a prompt with numbered
  context chunks and source citations, then calls the model.
- **analytics.py** -- `AnalyticsLLM` extends `BaseLLM` (default model:
  `gpt-5-mini`). Its `analyze_query` method returns structured JSON
  (topic, intent, category, complexity) for each user query.

### 5. Security (`src/security/`)

- **guardrails.py** -- Regex-based prompt injection detection
  (`check_injection`) plus a combined `validate_query` function that also
  enforces maximum query length. Intended as a first line of defence; a
  classifier-based approach can be layered on top.

### 6. Agent Core (`src/agent/`)

- **state.py** -- `AgentState`, a `TypedDict` that carries all data between
  graph nodes: the query, its embedding, memory results, web results, chunks,
  context, response, route taken, and timing.
- **nodes.py** -- Nine node / routing functions that each receive the full
  state and return a partial update:
  1. `embed_query_node` -- embed the user query.
  2. `vector_search_node` -- KNN search in Redis, decide hit or miss.
  3. `route_decision_node` -- conditional edge returning `"memory"` or
     `"web"`.
  4. `prepare_memory_context_node` -- build context from cached chunks.
  5. `check_sufficiency_node` -- uses the analytics LLM to verify the
     retrieved context actually answers the query.
  6. `sufficiency_route` -- conditional edge: `"sufficient"` proceeds to
     response generation; `"insufficient"` falls back to web search.
  7. `web_search_node` -- call Tavily.
  8. `store_web_results_node` -- chunk, embed, store, build context.
  9. `generate_response_node` -- call the conversation LLM.
  10. `log_interaction_node` -- record analytics and end time.
- **graph.py** -- Wires nodes into a LangGraph `StateGraph` with two
  conditional branches: one after `vector_search` (memory hit vs miss) and
  one after `check_sufficiency` (context answers the question vs does not).
  When cached context is found but deemed insufficient, the agent
  automatically falls back to live web search rather than generating a
  response from irrelevant context. All paths converge at
  `generate_response`, then flow through `log_interaction` to `END`.

### 7. CLI (`main.py`)

- Accepts a single positional query **or** an `--interactive` flag for a REPL
  loop.
- Validates input through the guardrails module before invoking the graph.
- Prints the route taken (memory hit / miss), similarity score, response,
  source URLs, and elapsed time.

### 8. Infrastructure

- **docker-compose.yml** -- Runs `redis/redis-stack:latest` which bundles
  RediSearch (required for vector indexing). Exposes port 6379 (Redis) and
  8001 (RedisInsight).
- **pyproject.toml / requirements.txt** -- All Python dependencies pinned to
  minimum compatible versions.
- **.env.example** -- Template for all required and optional environment
  variables.

### 9. Tests (`tests/`)

- **Unit tests** cover the chunker, guardrails, embedder (mocked), and Redis
  client (with a live-Redis marker).
- **Integration tests** exercise the full memory-hit and memory-miss flows
  end-to-end, requiring a running Redis Stack and valid API keys.

---

## Design Decisions

| Decision | Rationale |
|---|---|
| Tavily as sole search provider | Free tier (1 000 searches/month), returns pre-extracted content, purpose-built for LLM agents. |
| gpt-5.2 / gpt-5-mini defaults | Higher capability for user-facing answers; cheaper model for background analytics. Both are configurable. |
| Model-agnostic `BaseLLM` | Any OpenAI-compatible endpoint works. Changing models requires only an env var update. |
| Synchronous code | PoC simplicity. LangGraph invokes nodes sequentially; async adds complexity without PoC benefit. |
| Redis Stack (not plain Redis) | RediSearch module is required for `FT.CREATE` / `FT.SEARCH` vector operations. |
| FLAT index with COSINE distance | Simplest index type; suitable for the PoC scale (under 100 000 chunks). HNSW is a drop-in upgrade. |
| 0.7 similarity threshold | Balances precision and recall. Configurable via `SIMILARITY_THRESHOLD`. |

---

## How to Run

```bash
# 1. Start Redis Stack
docker compose up -d

# 2. Configure environment
cp .env.example .env
# fill in OPENAI_API_KEY and TAVILY_API_KEY

# 3. Install dependencies
uv sync          # or: pip install -r requirements.txt

# 4. Run a single query
python main.py "What is LangGraph?"

# 5. Interactive mode
python main.py --interactive

# 6. Run tests
pytest tests/unit -v
pytest tests/integration -v   # requires Redis + API keys
```

---

## What Is Not Included (Future Work)

- Async / parallel page fetching.
- HNSW index for large-scale deployment.
- Token-level cost tracking per query.
- Persistent conversation memory (multi-turn).
- Production hardening (rate limiting, circuit breakers, retries with backoff).
- Web UI or API server.
