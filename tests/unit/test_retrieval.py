from aurora.models import Document
from aurora.services.embeddings import VectorStore
from aurora.services.retrieval import RetrievalService


class TestRetrievalService:
    def test_retrieve_returns_chunks(self, vector_store: VectorStore, sample_documents: list[Document]):
        vector_store.ingest(sample_documents)
        service = RetrievalService(vector_store=vector_store, top_k=5, similarity_threshold=1.5)
        chunks = service.retrieve("private jet to Paris")
        assert len(chunks) > 0
        assert chunks[0].source_id is not None

    def test_threshold_filtering(self, vector_store: VectorStore, sample_documents: list[Document]):
        vector_store.ingest(sample_documents)
        strict = RetrievalService(vector_store=vector_store, top_k=5, similarity_threshold=0.01)
        chunks = strict.retrieve("completely unrelated quantum physics topic xyz123")
        lenient = RetrievalService(vector_store=vector_store, top_k=5, similarity_threshold=2.0)
        lenient_chunks = lenient.retrieve("completely unrelated quantum physics topic xyz123")
        assert len(chunks) <= len(lenient_chunks)

    def test_empty_store_returns_empty(self, vector_store: VectorStore):
        service = RetrievalService(vector_store=vector_store, top_k=5, similarity_threshold=1.5)
        chunks = service.retrieve("anything")
        assert len(chunks) == 0

    def test_chunks_have_required_fields(self, vector_store: VectorStore, sample_documents: list[Document]):
        vector_store.ingest(sample_documents)
        service = RetrievalService(vector_store=vector_store, top_k=5, similarity_threshold=1.5)
        chunks = service.retrieve("Paris")
        for chunk in chunks:
            assert chunk.source_id
            assert chunk.source_type
            assert chunk.text
            assert isinstance(chunk.distance, float)

    def test_respects_top_k(self, vector_store: VectorStore, sample_documents: list[Document]):
        vector_store.ingest(sample_documents)
        service = RetrievalService(vector_store=vector_store, top_k=2, similarity_threshold=2.0)
        chunks = service.retrieve("restaurant Paris booking")
        assert len(chunks) <= 2
