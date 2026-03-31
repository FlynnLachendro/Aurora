from aurora.models import Document
from aurora.services.embeddings import VectorStore


class TestVectorStore:
    def test_initially_empty(self, vector_store: VectorStore):
        assert vector_store.is_populated() is False

    def test_ingest_populates(self, vector_store: VectorStore, sample_documents: list[Document]):
        vector_store.ingest(sample_documents)
        assert vector_store.is_populated() is True
        assert vector_store.collection.count() == len(sample_documents)

    def test_upsert_idempotent(self, vector_store: VectorStore, sample_documents: list[Document]):
        vector_store.ingest(sample_documents)
        count_after_first = vector_store.collection.count()
        vector_store.ingest(sample_documents)
        count_after_second = vector_store.collection.count()
        assert count_after_first == count_after_second

    def test_query_returns_results(self, vector_store: VectorStore, sample_documents: list[Document]):
        vector_store.ingest(sample_documents)
        results = vector_store.query("private jet to Paris", top_k=3)
        assert results["ids"] is not None
        assert len(results["ids"][0]) > 0

    def test_query_ordering(self, vector_store: VectorStore, sample_documents: list[Document]):
        vector_store.ingest(sample_documents)
        results = vector_store.query("private jet to Paris", top_k=5)
        distances = results["distances"][0]
        assert distances == sorted(distances)

    def test_query_includes_metadata(self, vector_store: VectorStore, sample_documents: list[Document]):
        vector_store.ingest(sample_documents)
        results = vector_store.query("Paris restaurant", top_k=1)
        metadata = results["metadatas"][0][0]
        assert "source_id" in metadata
        assert "source_type" in metadata
        assert "user_name" in metadata

    def test_ingest_empty_list(self, vector_store: VectorStore):
        vector_store.ingest([])
        assert vector_store.is_populated() is False

    def test_query_empty_store(self, vector_store: VectorStore):
        results = vector_store.query("anything", top_k=5)
        assert len(results["ids"][0]) == 0
