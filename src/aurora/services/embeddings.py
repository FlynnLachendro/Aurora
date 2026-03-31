import chromadb
from loguru import logger

from aurora.core.constants import COLLECTION_NAME, UPSERT_BATCH_SIZE
from aurora.models import Document


class VectorStore:
    def __init__(self, persist_dir: str, embedding_model: str) -> None:
        self._client = chromadb.PersistentClient(path=persist_dir)
        self._embedding_model = embedding_model
        self._embedding_fn = chromadb.utils.embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
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
        if not documents:
            return

        for i in range(0, len(documents), UPSERT_BATCH_SIZE):
            batch = documents[i : i + UPSERT_BATCH_SIZE]
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
            logger.info(f"Upserted batch {i // UPSERT_BATCH_SIZE + 1} ({len(batch)} docs)")

        logger.info(f"Ingestion complete. Total: {self._collection.count()} documents")

    def query(
        self,
        text: str,
        top_k: int = 15,
        where: dict | None = None,
    ) -> dict:
        kwargs: dict = {
            "query_texts": [text],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        return self._collection.query(**kwargs)
