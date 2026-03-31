# Aurora Q&A Service

A question-answering service that provides precise, grounded insights into member data from Aurora's concierge platform. Answers natural language questions by combining semantic search over member history with LLM-powered answer generation.

## Architecture

```
Aurora API (Cloud Run)            Aurora Q&A Service (FastAPI)
┌──────────────────────┐          ┌─────────────────────────────────────┐
│ /messages/ (3,349)   │          │                                     │
│ /calendar-events/    │──fetch───│  ChromaDB          Gemini Flash     │
│ /spotify/            │  all at  │  (vector store) ──▶ (via OpenRouter) │
│ /whoop/              │  startup │                                     │
│ /hackathon/me/       │          │  POST /ask ──▶ retrieve ──▶ answer  │
└──────────────────────┘          └─────────────────────────────────────┘
```

**Data flow:**

1. **Startup**: Fetches all data from Aurora's API (messages, calendar, Spotify, Whoop, profile) concurrently
2. **Indexing**: Converts each record to natural language text, embeds via `all-MiniLM-L6-v2`, stores in ChromaDB
3. **Query**: Embeds the question, performs cosine similarity search for top-15 relevant chunks
4. **Answer**: Passes retrieved context to Gemini 2.0 Flash, which returns a structured answer with confidence score, source IDs, and reasoning trace

## Quick Start

### Local

```bash
# Install dependencies
uv sync --extra dev

# Set your OpenRouter API key
cp .env.example .env
# Edit .env with your key

# Run the server
uv run uvicorn aurora.main:app --reload
```

### Docker

```bash
docker build -t aurora-qa .
docker run -p 8000:8000 -e OPENROUTER_API_KEY=your-key aurora-qa
```

### Test it

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Sophia Al-Farsi'\''s most recent request?"}'
```

## API

### POST /ask

**Request:**
```json
{
  "question": "What is Amira's favorite restaurant in Paris?"
}
```

**Response:**
```json
{
  "answer": "Based on the available data, Amira has expressed a strong preference for Le Cinq in Paris.",
  "confidence": 0.85,
  "sources": ["msg-003", "msg-001"],
  "metadata": {
    "reasoning": "Found explicit positive mention of Le Cinq in msg-003 where the member said they 'loved dining' there and requested a repeat booking. msg-001 confirms Paris as a frequent destination.",
    "sources_considered": 15,
    "retrieval_time_ms": 12.5,
    "generation_time_ms": 650.0
  }
}
```

### GET /health

Returns `{"status": "ok"}` for health checks.

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Vector store | Single ChromaDB collection with `source_type` metadata | Cross-domain questions need unified search across all data types |
| Embedding model | all-MiniLM-L6-v2 (384-dim) | Fast, CPU-only, good quality, ~80MB — no GPU needed |
| Document format | Natural language sentences | Embedding model was trained on natural language, not raw JSON |
| LLM | Gemini 2.0 Flash via OpenRouter | Fast inference (~500-800ms), cost-effective, JSON output mode |
| No-data handling | Similarity threshold filter, skip LLM call | Prevents hallucination on out-of-scope queries, saves latency |
| Warm restart | `is_populated()` check on startup | Skip re-ingestion if data exists — fast restarts after first deploy |
| Confidence calibration | Rubric in system prompt (0.9+ explicit, 0.7-0.9 inference, <0.5 insufficient) | Prevents model from always returning high confidence |

## Testing

```bash
# Run all tests (55 tests)
uv run pytest tests/ -v

# Linting and type checking
uv run ruff format .
uv run ruff check . --fix
uv run mypy src/aurora
```

## Production Readiness

If scaling to 100,000 members with 10 years of history each, the first architectural change would be **migrating from a single in-memory vector store to a distributed, member-partitioned retrieval system**.

### What changes at scale

The current architecture ingests ~3,900 documents into a single ChromaDB collection at startup. At 100K members with 10 years of dense conversation history, we'd be looking at hundreds of millions of documents. This breaks the current design in three ways: startup ingestion becomes impractical (hours, not seconds), a single vector index can't serve sub-second queries over that volume, and the entire dataset doesn't fit in memory on a single node.

### The first change: member-scoped retrieval with incremental ingestion

1. **Member-partitioned vector store**: Replace the single ChromaDB collection with a distributed vector database (Qdrant or Weaviate) partitioned by `member_id`. Each query scopes to a single member's partition, keeping search fast regardless of total data volume. This also enables tenant isolation and GDPR-compliant deletion (drop a member's partition).

2. **Event-driven incremental ingestion**: Replace startup bulk-fetch with a message queue (SQS/Kafka). As new messages arrive in Aurora's system, they're published to the queue, embedded, and upserted into the relevant member partition. No more full re-indexing.

3. **Two-stage retrieval**: At millions of documents per member, approximate nearest neighbor (ANN) search may surface false positives. Add a cross-encoder re-ranking stage: ANN retrieves top-50 candidates, a cross-encoder (e.g., `ms-marco-MiniLM`) re-scores them, and only the top-10 go to the LLM. This adds ~100ms but significantly improves precision.

4. **Response caching**: Many concierge queries are repeated ("What's my usual hotel in Tokyo?"). A semantic cache (hash the question embedding, TTL-based invalidation when new messages arrive for that member) would eliminate redundant LLM calls for common patterns.

5. **Model routing**: Route simple factual lookups ("What's my next meeting?") to a smaller/faster model, and complex analytical queries ("How has my sleep quality trended this month?") to a more capable one. Reduces average latency and cost.
