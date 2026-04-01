"""
Retrieval service — multi-source semantic search.

The key challenge: 3,349 concierge messages dominate the vector space, drowning out
31 whoop records, 154 calendar events, and 338 spotify streams. A single top-K query
would return only messages for most questions.

Solution: embed the question once, then run a primary unfiltered query PLUS per-source-type
enrichment queries with metadata filters. This ensures rare data types always get
representation in the results. All 5 queries reuse the same embedding vector (~200ms
for 1 API call + ~20ms for 5 local ChromaDB searches).
"""

from loguru import logger

from aurora.core.constants import (
    SOURCE_TYPE_CALENDAR,
    SOURCE_TYPE_PROFILE,
    SOURCE_TYPE_SPOTIFY,
    SOURCE_TYPE_WHOOP,
)
from aurora.models import RetrievedChunk
from aurora.services.embeddings import VectorStore

# Non-message source types that need enrichment queries to avoid being drowned out
ENRICHMENT_TYPES = [SOURCE_TYPE_CALENDAR, SOURCE_TYPE_SPOTIFY, SOURCE_TYPE_WHOOP, SOURCE_TYPE_PROFILE]


class RetrievalService:
    def __init__(self, vector_store: VectorStore, top_k: int, similarity_threshold: float) -> None:
        self._vector_store = vector_store
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    def _parse_results(self, raw: dict) -> list[RetrievedChunk]:
        """Convert ChromaDB query results into typed RetrievedChunk objects.

        Filters out results above the similarity threshold (cosine distance).
        Higher distance = less similar. Threshold of 1.2 is lenient — we let the
        LLM decide relevance rather than being too aggressive with filtering.
        """
        chunks: list[RetrievedChunk] = []
        if not raw["ids"] or not raw["ids"][0]:
            return chunks

        for i, doc_id in enumerate(raw["ids"][0]):
            distance = raw["distances"][0][i] if raw["distances"] else 0.0
            if distance > self._similarity_threshold:
                continue

            metadata = raw["metadatas"][0][i] if raw["metadatas"] else {}
            text = raw["documents"][0][i] if raw["documents"] else ""

            chunks.append(
                RetrievedChunk(
                    source_id=metadata.get("source_id", doc_id),
                    source_type=metadata.get("source_type", ""),
                    text=text,
                    distance=distance,
                    user_name=metadata.get("user_name", ""),
                    timestamp=metadata.get("timestamp", ""),
                )
            )
        return chunks

    def retrieve(self, question: str) -> list[RetrievedChunk]:
        """Retrieve relevant chunks using embed-once + multi-source strategy.

        1. Embed question once via Gemini API (~200ms)
        2. Primary query: top-K across all sources (~5ms, reuses vector)
        3. Enrichment queries: top-5 per source type with metadata filter (~5ms each)
        4. Deduplicate and sort by distance
        """
        # Embed question ONCE, reuse vector for all queries
        question_embedding = self._vector_store.embed(question)

        # Primary query: top results across all sources
        primary = self._vector_store.query(embedding=question_embedding, top_k=self._top_k)
        chunks = self._parse_results(primary)
        seen_ids = {c.source_id for c in chunks}

        # Enrichment: query each non-message source type separately to ensure
        # rare types (whoop, calendar, spotify) aren't drowned out by 3,000+ messages
        for source_type in ENRICHMENT_TYPES:
            enrichment = self._vector_store.query(
                embedding=question_embedding,
                top_k=5,
                where={"source_type": source_type},
            )
            for chunk in self._parse_results(enrichment):
                if chunk.source_id not in seen_ids:
                    chunks.append(chunk)
                    seen_ids.add(chunk.source_id)

        # Sort all chunks by distance (best first)
        chunks.sort(key=lambda c: c.distance)

        logger.info(f"Retrieved {len(chunks)} chunks ({len(seen_ids)} unique, threshold={self._similarity_threshold})")
        return chunks
