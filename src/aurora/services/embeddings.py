"""ChromaDB vector store — single collection, Gemini embedding API, cosine similarity."""

import time
from typing import Any

import chromadb
import chromadb.utils.embedding_functions as ef
from loguru import logger

from aurora.core.constants import COLLECTION_NAME, UPSERT_BATCH_SIZE
from aurora.models import Document


class VectorStore:
    def __init__(self, persist_dir: str, embedding_model: str, use_gemini: bool = False) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        # Gemini API in prod, ChromaDB default in tests (no API key needed)
        embedding_fn: Any
        if use_gemini:
            embedding_fn = ef.GoogleGeminiEmbeddingFunction(
                model_name=embedding_model,
            )
        else:
            embedding_fn = ef.DefaultEmbeddingFunction()
        self._embedding_fn = embedding_fn
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
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
        """Batched upsert with retry for Google's 429 rate limits. Batch size 100."""
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

    def embed(self, text: str) -> list[Any]:
        """Embed once, return raw vector — reused across all 5 retrieval queries."""
        return self._embedding_fn([text])[0]  # type: ignore[no-any-return]

    def query(
        self,
        text: str | None = None,
        embedding: list[Any] | None = None,
        top_k: int = 15,
        where: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Query by text or pre-computed vector. Pre-computed skips the embed API call."""
        kwargs: dict[str, Any] = {
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if embedding is not None:
            kwargs["query_embeddings"] = [embedding]
        elif text is not None:
            kwargs["query_texts"] = [text]
        if where:
            kwargs["where"] = where
        return dict(self._collection.query(**kwargs))
