"""Multi-source retrieval — embed once, query primary + per-source-type enrichment."""

from typing import Any

from loguru import logger

from aurora.core.constants import (
    SOURCE_TYPE_CALENDAR,
    SOURCE_TYPE_PROFILE,
    SOURCE_TYPE_SPOTIFY,
    SOURCE_TYPE_WHOOP,
)
from aurora.models import RetrievedChunk
from aurora.services.embeddings import VectorStore

# Rare source types that get drowned out by 3,349 messages without enrichment queries
ENRICHMENT_TYPES = [SOURCE_TYPE_CALENDAR, SOURCE_TYPE_SPOTIFY, SOURCE_TYPE_WHOOP, SOURCE_TYPE_PROFILE]


class RetrievalService:
    def __init__(self, vector_store: VectorStore, top_k: int, similarity_threshold: float) -> None:
        self._vector_store = vector_store
        self._top_k = top_k
        self._similarity_threshold = similarity_threshold

    def _parse_results(self, raw: dict[str, Any]) -> list[RetrievedChunk]:
        """Parse ChromaDB results, filtering by cosine distance threshold."""
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
        """Embed once → primary query → per-source enrichment → dedup → sort."""
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
