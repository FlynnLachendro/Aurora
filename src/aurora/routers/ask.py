"""
POST /ask endpoint — the core API surface.

Orchestrates retrieval → LLM generation and adds timing metadata to the response.
Services are accessed via request.app.state (set during lifespan startup),
making them easy to override in integration tests.
"""

import time

from fastapi import APIRouter, Request
from loguru import logger

from aurora.models import AskRequest, AskResponse
from aurora.services.llm import LLMService
from aurora.services.retrieval import RetrievalService

router = APIRouter()


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    retrieval_service: RetrievalService = request.app.state.retrieval_service
    llm_service: LLMService = request.app.state.llm_service
    profile = request.app.state.profile

    logger.info(f"Question: {body.question}")

    # Time retrieval (embed + ChromaDB search) and generation (LLM call) separately
    # so metadata shows where latency is spent — useful for debugging and evaluation.
    retrieval_start = time.perf_counter()
    chunks = retrieval_service.retrieve(body.question)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    generation_start = time.perf_counter()
    response = await llm_service.generate_answer(body.question, chunks, profile)
    generation_ms = (time.perf_counter() - generation_start) * 1000

    response.metadata.retrieval_time_ms = round(retrieval_ms, 2)
    response.metadata.generation_time_ms = round(generation_ms, 2)

    logger.info(
        f"Answer generated — confidence={response.confidence}, "
        f"sources={len(response.sources)}, "
        f"retrieval={retrieval_ms:.0f}ms, generation={generation_ms:.0f}ms"
    )

    return response
