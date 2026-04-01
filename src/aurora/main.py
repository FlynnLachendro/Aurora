"""FastAPI app — lifespan fetches Aurora data, embeds into ChromaDB, wires up services."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from loguru import logger

from aurora.config import Settings
from aurora.core.logging import setup_logging
from aurora.routers.ask import router as ask_router
from aurora.services.embeddings import VectorStore
from aurora.services.ingestion import fetch_all_data
from aurora.services.llm import LLMService
from aurora.services.retrieval import RetrievalService


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    setup_logging()
    settings = Settings()

    # ChromaDB reads GEMINI_API_KEY from os.environ, not constructor args
    if settings.gemini_api_key:
        import os

        os.environ["GEMINI_API_KEY"] = settings.gemini_api_key

    logger.info("Initializing vector store...")
    vector_store = VectorStore(
        persist_dir=settings.chroma_persist_dir,
        embedding_model=settings.embedding_model,
        use_gemini=bool(settings.gemini_api_key),
    )

    # Warm restart: skip ingestion. Cold start: fetch + embed ~3,873 docs.
    if vector_store.is_populated():
        logger.info("Vector store already populated, skipping ingestion")
    else:
        logger.info("Fetching data from Aurora API...")
        documents, profile = await fetch_all_data(settings)
        logger.info("Ingesting documents into vector store...")
        vector_store.ingest(documents)
        app.state.profile = profile

    # On warm restart, profile needs a separate fetch (single GET)
    if not hasattr(app.state, "profile") or app.state.profile is None:
        import httpx

        from aurora.services.ingestion import fetch_profile

        async with httpx.AsyncClient(
            base_url=settings.aurora_api_base_url, timeout=30.0, follow_redirects=True
        ) as client:
            from aurora.models import UserProfile

            profile_data = await fetch_profile(client)
            app.state.profile = UserProfile(**profile_data)

    # Services on app.state for DI via request.app.state in route handlers
    app.state.retrieval_service = RetrievalService(
        vector_store=vector_store,
        top_k=settings.retrieval_top_k,
        similarity_threshold=settings.similarity_threshold,
    )

    app.state.llm_service = LLMService(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
        model=settings.llm_model,
    )

    logger.info("Aurora Q&A service ready")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Aurora Q&A Service",
    description="Answers natural language questions grounded in member data",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(ask_router)


@app.get("/health")
async def health() -> dict[str, str]:
    """Railway health check."""
    return {"status": "ok"}
