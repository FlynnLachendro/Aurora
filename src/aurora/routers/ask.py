"""POST /ask — orchestrates retrieval → LLM generation, adds timing metadata."""

import time

from fastapi import APIRouter, Request
from loguru import logger

from aurora.models import AskRequest, AskResponse, RetrievedChunk
from aurora.services.llm import LLMService
from aurora.services.retrieval import RetrievalService

router = APIRouter()

# Benchmarked caps 1-10: confidence flat (0.79-0.83), gen time scales linearly.
# Cap=2 had highest avg confidence (0.83) at near-fastest speed.
MAX_LLM_CHUNKS = 2


def retrieval_confidence(chunks: list[RetrievedChunk]) -> float:
    """Compute confidence from cosine distances. Lower distance = higher confidence.

    Cosine distance range: 0 (identical) to 2 (opposite).
    We map the best chunk's distance to a 0-1 confidence score:
    distance 0.0 → 1.0, distance 1.0 → 0.0, distance ≥1.0 → 0.0.
    """
    if not chunks:
        return 0.0
    best_distance = chunks[0].distance  # Already sorted by distance
    return max(0.0, min(1.0, 1.0 - best_distance))


def hybrid_confidence(llm_confidence: float, chunks: list[RetrievedChunk]) -> float:
    """Blend LLM self-reported confidence with retrieval-based confidence.

    50/50 weight: LLM judges answer quality, retrieval grounds it in data relevance.
    If the LLM is overconfident but retrieval was weak, the score gets pulled down.
    Exception: if the LLM says 0.0 (no data), trust it — don't let retrieval inflate.
    """
    if llm_confidence == 0.0:
        return 0.0
    ret_conf = retrieval_confidence(chunks)
    return round(0.5 * llm_confidence + 0.5 * ret_conf, 2)


@router.post("/ask", response_model=AskResponse)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    retrieval_service: RetrievalService = request.app.state.retrieval_service
    llm_service: LLMService = request.app.state.llm_service
    profile = request.app.state.profile

    logger.info(f"Question: {body.question}")

    # Time retrieval and generation separately for observability
    retrieval_start = time.perf_counter()
    chunks = retrieval_service.retrieve(body.question)
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    # Send only the best chunks to the LLM — already sorted by distance
    cap = body.max_chunks or MAX_LLM_CHUNKS
    llm_chunks = chunks[:cap]

    generation_start = time.perf_counter()
    response = await llm_service.generate_answer(body.question, llm_chunks, profile)
    generation_ms = (time.perf_counter() - generation_start) * 1000

    # Hybrid confidence: blend LLM self-report with retrieval distance signal
    response.confidence = hybrid_confidence(response.confidence, llm_chunks)

    # Opt-in judge: used during development to validate confidence scoring. Off by default.
    if body.judge and llm_chunks:
        judge_result = await llm_service.judge_answer(body.question, response.answer, llm_chunks)
        response.metadata.judge = judge_result

    response.metadata.retrieval_time_ms = round(retrieval_ms, 2)
    response.metadata.generation_time_ms = round(generation_ms, 2)

    logger.info(
        f"Answer generated — confidence={response.confidence}, "
        f"sources={len(response.sources)}, "
        f"retrieval={retrieval_ms:.0f}ms, generation={generation_ms:.0f}ms"
    )

    return response
