# Napkin

## Corrections
| Date | Source | What Went Wrong | What To Do Instead |
|------|--------|----------------|-------------------|
| 2026-03-31 | self | httpx got 405 on Aurora API due to redirect | Always use `follow_redirects=True` with httpx.AsyncClient |
| 2026-03-31 | self | Aurora API returned 400 intermittently at certain skip values | Added `_get_with_retry()` with 3 attempts and backoff |
| 2026-03-31 | self | Whoop/calendar/spotify data drowned out by 3,349 messages in single vector query | Multi-source retrieval: primary query + per-source-type enrichment queries |
| 2026-03-31 | self | Integration tests failed because ASGITransport doesn't trigger FastAPI lifespan | Set app.state directly in test fixture instead of relying on lifespan |
| 2026-03-31 | self | sentence-transformers embedding on Railway CPU took 13s per query | Swap to API-based embeddings (Gemini embedding API) — network is faster than weak CPU |
| 2026-03-31 | self | GoogleGeminiEmbeddingFunction reads GEMINI_API_KEY from os.environ, not constructor | Must set os.environ["GEMINI_API_KEY"] before creating the embedding function |
| 2026-03-31 | self | Google embed API has 100-item batch limit, also hit 3000 req/min rate limit | Reduced UPSERT_BATCH_SIZE to 100, added retry with backoff for 429s |
| 2026-03-31 | self | Railway PORT env var not used — hardcoded 8000 in Dockerfile CMD | Use `${PORT}` in CMD with shell form |

## User Preferences
- Uses `uv` for package management, avoids venvs
- Concise commit messages, no Co-Authored-By lines
- Wait for explicit approval before commits, pushes, PRs
- No legacy wrappers — update callers directly when refactoring
- Test-driven: write tests alongside features, not after
- Type everything — mypy strict, catches bugs before runtime
- Logging baked in from day one with loguru (INFO happy path, DEBUG detail)
- Format/lint on save — no style debates (ruff format + ruff check)

## Patterns That Work
- FastAPI + Python 3.11+ is Flynn's strongest stack — default for all backends
- SQLAlchemy 2.0 async + Supabase Postgres for DB layer
- Pydantic models for request/response validation
- Backend tooling: pytest + pytest-asyncio, mypy, ruff, loguru
- Docker with Python slim + uv for containerized deployments
- Railway for backend hosting ($5/mo, scales from hobby to paid)
- SSE via sse-starlette for streaming (used in Arbor)
- OpenRouter for multi-model LLM access (DeepSeek, Gemini, GPT)
- PostHog for analytics
- pyproject.toml sections for [tool.pytest], [tool.mypy], [tool.ruff]
- Multi-source retrieval: primary unfiltered query + per-source-type enrichment queries to prevent rare types being drowned out
- ChromaDB `where` metadata filters for per-source-type queries
- Retry with backoff for flaky external APIs (`_get_with_retry`)

## Patterns That Don't Work
- Single ChromaDB query across mixed-size source types — 3,349 messages drown out 31 whoop records
- Anthropic direct API with Flynn's current API key — only has access to old models (claude-3-haiku), not Sonnet 4.5+
- Anthropic doesn't expose OpenAI-compatible chat/completions endpoint — must use native `anthropic` SDK
- `response_format={"type": "json_object"}` not supported on Anthropic API (needs `json_schema`)
- `gemini-2.0-flash` deprecated on Google direct API for new users — use `gemini-2.5-flash`

## LLM Benchmarks (2026-03-31, from local machine)
| Provider | Model | Latency | Confidence | Notes |
|----------|-------|---------|------------|-------|
| OpenRouter | gemini-2.0-flash-001 | ~1,400-1,600ms | 0.7 | Fastest, good enough quality |
| Google direct | gemini-2.5-flash | ~2,300-3,300ms | 0.8-0.9 | Better reasoning but too slow for <2s target |
| Anthropic direct | Sonnet 4.5 | N/A | N/A | 404 — key lacks access to newer models |
- Chose Gemini Flash via OpenRouter: fastest, comfortably under 2s, good quality
- OpenRouter middleman overhead (~50-200ms) is small vs model speed differences

## Chunk Cap Sweep (2026-04-01, against live Railway deployment)
- Swept max_chunks 1-10 across 13 test questions (all data types + edge cases)
- Confidence is FLAT (0.79-0.83) regardless of chunk count — top 1-2 chunks contain the signal
- Generation time scales linearly: cap=1 → 1,141ms, cap=10 → 2,473ms
- Cap=2 had highest avg confidence (0.83) — set as default
- No-data handling perfect at all cap values (2/2 out-of-scope questions correctly returned 0.0)
- Key insight: more context ≠ better answers for this dataset. Retrieval quality matters more than quantity.

## Domain Notes

### Project: Aurora Take-Home
- **What**: AI/ML Engineer take-home assignment for Aurora (luxury concierge company)
- **Task**: Build a Q&A service over member message history (POST /ask endpoint)
- **Core requirement**: Natural language question → structured response with answer, confidence, sources (msg IDs), reasoning trace
- **Constraints**: <2s latency, Python any framework, must be publicly deployed
- **Eval criteria**: Precision (correct answers from dense history), Reliability (no-data/ambiguous handling), Traceability (reasoning field)
- **Submission**: Public GitHub repo + README (architecture + "Production Readiness" section on scaling to 100K members w/ 10yr history) + Live URL
- **Integration**: Must work with Aurora's Messages API for member data
- **Similar prior work**: Flynn did a Sante (blood test analysis) take-home before — technical assessment pattern is familiar

### Flynn's Background (relevant for this task)
- PhD in Biomedical Engineering
- AI engineer — multi-agent systems, LLM evaluation, AI safety
- Founding Engineer at Nur Opus (building Prism)
- Has shipped multiple production apps (Arbor, Lattice, Layers, Askaroo)
- Deep experience with LLM pipelines, RAG patterns, and API design
