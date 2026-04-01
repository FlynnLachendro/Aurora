"""
Vector store service wrapping ChromaDB.

Uses a single collection with source_type metadata tags so cross-domain questions
("What was James doing on Feb 15?") can retrieve calendar + whoop + spotify results
in one search. Embedding is done via Gemini's API (fast, no local model needed),
with a fallback to ChromaDB's default local embedder for tests.
"""

import time

import chromadb
import chromadb.utils.embedding_functions as ef
from loguru import logger

from aurora.core.constants import COLLECTION_NAME, UPSERT_BATCH_SIZE
from aurora.models import Document


class VectorStore:
    def __init__(self, persist_dir: str, embedding_model: str, use_gemini: bool = False) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        # In production, use Gemini's embedding API (fast, ~200ms per call).
        # In tests, fall back to ChromaDB's default local embedder (no API key needed).
        if use_gemini:
            self._embedding_fn = ef.GoogleGeminiEmbeddingFunction(
                model_name=embedding_model,
            )
        else:
            self._embedding_fn = ef.DefaultEmbeddingFunction()
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def collection(self) -> chromadb.Collection:
        return self._collection

    def is_populated(self) -> bool:
        count = self._collection.count()
        logger.info(f"Collection '{COLLECTION_NAME}' has {count} documents")
        return count > 0

    def ingest(self, documents: list[Document]) -> None:
        """Embed and upsert documents into ChromaDB in batches.

        Uses upsert (not add) for idempotency — safe to re-run without duplicates.
        Includes retry with backoff for Google's embedding API rate limits (429s).
        Batch size is 100 to stay within Google's per-request limit.
        """
        if not documents:
            return

        for i in range(0, len(documents), UPSERT_BATCH_SIZE):
            batch = documents[i : i + UPSERT_BATCH_SIZE]
            batch_num = i // UPSERT_BATCH_SIZE + 1
            for attempt in range(5):
                try:
                    self._collection.upsert(
                        ids=[doc.source_id for doc in batch],
                        documents=[doc.text for doc in batch],
                        metadatas=[
                            {
                                "source_id": doc.source_id,
                                "source_type": doc.source_type,
                                "user_name": doc.user_name,
                                "timestamp": doc.timestamp,
                            }
                            for doc in batch
                        ],
                    )
                    break
                except ValueError as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        wait = 15 * (attempt + 1)
                        logger.warning(f"Rate limited on batch {batch_num}, waiting {wait}s...")
                        time.sleep(wait)
                    else:
                        raise
            logger.info(f"Upserted batch {batch_num} ({len(batch)} docs)")

        logger.info(f"Ingestion complete. Total: {self._collection.count()} documents")

    def embed(self, text: str) -> list[float]:
        """Embed a single text string and return the raw vector.

        Used by RetrievalService to embed the question ONCE and reuse the vector
        across all 5 retrieval queries (1 primary + 4 enrichment), avoiding
        redundant API calls.
        """
        return self._embedding_fn([text])[0]

    def query(
        self,
        text: str | None = None,
        embedding: list[float] | None = None,
        top_k: int = 15,
        where: dict | None = None,
    ) -> dict:
        """Search ChromaDB by text (embeds it) or pre-computed embedding vector.

        Passing a pre-computed embedding skips the embedding API call entirely,
        making subsequent queries after the first near-instant (~5ms local search).
        """
        kwargs: dict = {
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if embedding is not None:
            kwargs["query_embeddings"] = [embedding]
        elif text is not None:
            kwargs["query_texts"] = [text]
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)
