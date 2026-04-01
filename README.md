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
| Embedding | Gemini Embedding API (`gemini-embedding-001`) | API-based embedding is faster on cloud infra than running a local model on limited CPU |
| Embedding strategy | Embed question once, reuse vector for all 5 retrieval queries | Avoids redundant API calls — 1 embed call instead of 5 |
| Retrieval | Multi-source enrichment (primary + per-source-type queries) | Prevents rare data types (31 whoop records) from being drowned out by 3,349 messages |
| Document format | Natural language sentences | Embedding models are trained on natural language, not raw JSON |
| LLM | Gemini 2.0 Flash via OpenRouter | Best balance of speed, quality, and reliability (see benchmarks below) |
| No-data handling | Similarity threshold filter, skip LLM call | Prevents hallucination on out-of-scope queries, saves latency |
| LLM context cap | Top 2 chunks sent to LLM (see chunk optimization below) | Benchmarked 1-10: confidence flat, generation time scales linearly |
| Warm restart | `is_populated()` check on startup | Skip re-ingestion if data exists — fast restarts after first deploy |
| Confidence scoring | Hybrid: 50% retrieval distance + 50% LLM self-report | Grounds confidence in data relevance, not just LLM self-assessment (see below) |

## Confidence Scoring

Rather than relying solely on the LLM's self-reported confidence (which can be poorly calibrated), we use a **hybrid approach** that blends two signals:

1. **Retrieval confidence** (50%) — derived from the cosine distance of the best matching chunk. Distance 0.0 (identical) maps to 1.0 confidence, distance 1.0+ maps to 0.0. This is grounded in actual data similarity.
2. **LLM confidence** (50%) — the model's self-assessed score, guided by a calibration rubric in the system prompt (0.9+ = explicit, 0.7-0.9 = inference, <0.5 = insufficient).

`final_confidence = 0.5 * retrieval_confidence + 0.5 * llm_confidence`

This means even if the LLM is overconfident about a weakly-matched retrieval, the low retrieval score pulls the final confidence down. Conversely, if retrieval finds a strong match but the LLM hedges, the retrieval signal lifts the score.

For no-data scenarios (no chunks pass threshold or LLM returns 0.0 confidence), confidence returns 0.0 — the retrieval signal is not allowed to inflate a no-data result.

### LLM-as-Judge Verification

To validate this approach, we implemented an optional independent judge (`judge: true` in the request body). A separate LLM call evaluates whether the generated answer is factually supported by the source data, without seeing the original question-answering prompt.

| Query type | Hybrid confidence | Judge score | Judge agrees? |
|---|---|---|---|
| Factual | 0.89 | 1.0 | Yes |
| Health/Whoop | 0.83 | 1.0 | Yes |
| Calendar | 0.85 | 1.0 | Yes |
| Music/Spotify | 0.85 | 1.0 | Yes |
| Profile | 0.89 | 1.0 | Yes |
| No data | 0.00 | 1.0 | Yes |
| Allergy | 0.88 | 1.0 | Yes |
| Temporal | 0.90 | 1.0 | Yes |

The judge confirms all answers are fully grounded in source data (1.0 across the board), including the no-data case where "I can't answer this" is correctly recognized as a valid grounded response. The hybrid confidence scores (0.83-0.90) are intentionally more conservative — the retrieval distance signal adds appropriate caution for a concierge service where over-confidence has real consequences.

## LLM Benchmarks

We evaluated 5 models via OpenRouter across 5 test questions (factual, health/whoop, calendar, preference, and out-of-scope). Each model was scored on average latency (generation only), average confidence calibration, and reliability (successful JSON responses).

| Model | Avg Latency | Avg Confidence | Reliability | Notes |
|-------|------------|----------------|-------------|-------|
| **Gemini 2.0 Flash** | **1,128ms** | **0.70** | **5/5** | Best balance — fast, reliable, well-calibrated |
| Gemini 2.0 Flash Lite | 1,119ms | 0.66 | 5/5 | Marginally faster, slightly less precise on nuanced questions |
| Claude 3.5 Haiku | 2,469ms | 0.54 | 5/5 | Overly cautious (0.10 confidence on answerable questions), hallucinated on out-of-scope query |
| Llama 4 Scout | 2,868ms | 0.65 | 4/5 | JSON output failed on 1/5 queries |
| Qwen 3 8B | 6,234ms | 0.94 | 4/5 | Highest confidence but 5x slower, JSON parsing unreliable |

**Selected: Gemini 2.0 Flash** — fastest reliable model with well-calibrated confidence scores. Correctly returns 0.0 confidence on out-of-scope questions (no hallucination), while maintaining high confidence (0.9-1.0) on directly answerable queries.

## Chunk Cap Optimization

After selecting the LLM, we swept the number of context chunks (1-10) sent to the model across 13 test questions covering all data types, edge cases, and no-data scenarios. The retrieval layer finds 20-30 chunks for diversity, but only the top N (by cosine distance) are passed to the LLM.

| Chunks | Avg Confidence | Avg Gen Time | Avg Total | No-data correct |
|--------|---------------|-------------|-----------|-----------------|
| **1** | **0.80** | **1,141ms** | **2,933ms** | **2/2** |
| **2** | **0.83** | **1,260ms** | **3,211ms** | **2/2** |
| 3 | 0.82 | 1,517ms | 3,211ms | 2/2 |
| 4 | 0.79 | 1,761ms | 3,149ms | 2/2 |
| 5 | 0.80 | 1,779ms | 3,209ms | 2/2 |
| 6 | 0.82 | 1,845ms | 3,313ms | 2/2 |
| 7 | 0.81 | 1,927ms | 3,803ms | 2/2 |
| 8 | 0.80 | 2,259ms | 4,073ms | 2/2 |
| 9 | 0.80 | 2,307ms | 3,954ms | 2/2 |
| 10 | 0.82 | 2,473ms | 4,170ms | 2/2 |

**Finding: confidence is flat (0.79-0.83) across all chunk counts, while generation time scales linearly.** The top 1-2 chunks contain the relevant signal; additional chunks add tokens without improving answer quality. No-data handling (out-of-scope questions returning 0.0 confidence) is perfect across all values.

**Selected: 2 chunks** — highest average confidence (0.83) at nearly the fastest generation time. Gives the LLM a second data point to cross-reference without the cost of additional context.

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
