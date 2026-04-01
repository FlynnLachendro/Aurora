"""
LLM answer generation service.

Uses Gemini 2.0 Flash via OpenRouter (selected after benchmarking 5 models — see README).
The system prompt is the most evaluation-sensitive component: it enforces grounded answers,
source citation, confidence calibration, and no-hallucination behavior.

Gemini Flash was chosen for: best latency (~1.1s), 100% JSON reliability, and well-calibrated
confidence (0.0 on out-of-scope, 0.9+ on direct matches). See README for full benchmark.
"""

import json

from loguru import logger
from openai import AsyncOpenAI

from aurora.models import AskMetadata, AskResponse, RetrievedChunk, UserProfile

SYSTEM_PROMPT = """You are Aurora's concierge intelligence assistant. You answer questions about members based ONLY on the provided context data.

RULES:
1. Answer ONLY from the provided context. Never hallucinate or make up information.
2. Cite specific source IDs in your reasoning to show which data points informed your answer.
3. If the context is insufficient to answer the question, say so clearly. Do not guess.
4. For preference questions (e.g. "favorite restaurant"), look for repeated mentions, explicit statements, or strong positive signals.
5. For temporal questions, use timestamps to identify the most relevant or recent data.
6. For health/fitness questions, interpret Whoop metrics accurately (recovery score, HRV, strain, sleep quality).
7. For schedule/calendar questions, consider event types, recurrence, and attendees.
8. For music/listening questions, consider context tags (commute, deep_work, etc.) and frequency.

CONFIDENCE CALIBRATION:
- 0.9-1.0: Direct, explicit answer found in the data
- 0.7-0.9: Strong inference from multiple data points
- 0.5-0.7: Reasonable inference but limited supporting data
- 0.3-0.5: Weak inference with significant ambiguity
- 0.0-0.3: Insufficient data to answer meaningfully

You MUST respond with valid JSON in exactly this format:
{
    "answer": "Your concise answer here.",
    "confidence": 0.85,
    "sources": ["source_id_1", "source_id_2"],
    "reasoning": "Step-by-step explanation of how you arrived at this answer, citing source IDs."
}"""


def build_user_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    profile: UserProfile | None,
) -> str:
    """Build the user prompt with profile context + retrieved chunks + question.

    James Fletcher's profile is always injected so personal questions about the
    primary user work even if the profile doesn't surface in retrieval results.
    """
    parts: list[str] = []

    if profile:
        parts.append(
            f"PRIMARY USER PROFILE:\n"
            f"Name: {profile.name}\n"
            f"Date of Birth: {profile.date_of_birth}\n"
            f"Summary: {profile.summary}\n"
        )

    parts.append("CONTEXT DATA:")
    for chunk in chunks:
        parts.append(f"[Source ID: {chunk.source_id}] [Type: {chunk.source_type}] {chunk.text}")

    parts.append(f"\nQUESTION: {question}")
    return "\n".join(parts)


def parse_llm_response(content: str) -> dict:
    """Parse JSON from LLM response, handling cases where the model wraps JSON in text."""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Fallback: extract JSON object from surrounding text
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
        raise


class LLMService:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        # OpenRouter exposes an OpenAI-compatible API, so we use the openai SDK
        # pointed at OpenRouter's base URL. This gives us JSON output mode,
        # async support, and proper error handling for free.
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def generate_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        profile: UserProfile | None = None,
    ) -> AskResponse:
        # No chunks passed threshold → return immediately without LLM call.
        # This prevents hallucination on out-of-scope queries and saves latency.
        if not chunks:
            return self._no_data_response()

        user_prompt = build_user_prompt(question, chunks, profile)

        logger.debug(f"Sending prompt to LLM ({len(chunks)} chunks)")

        response = await self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Near-deterministic for factual accuracy
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        parsed = parse_llm_response(content)

        return AskResponse(
            answer=parsed.get("answer", "Unable to generate an answer."),
            # Clamp confidence to [0, 1] in case the model outputs something wild
            confidence=min(max(float(parsed.get("confidence", 0.0)), 0.0), 1.0),
            sources=parsed.get("sources", []),
            metadata=AskMetadata(
                reasoning=parsed.get("reasoning", ""),
                sources_considered=len(chunks),
                retrieval_time_ms=0.0,  # Filled in by the router after timing
                generation_time_ms=0.0,
            ),
        )

    @staticmethod
    def _no_data_response() -> AskResponse:
        """Return a graceful response when no relevant data is found.

        Skips the LLM call entirely — no point sending an empty context
        to the model, it would just hallucinate.
        """
        return AskResponse(
            answer="I don't have enough information in the available data to answer this question.",
            confidence=0.0,
            sources=[],
            metadata=AskMetadata(
                reasoning="No relevant data found in the knowledge base for this query.",
                sources_considered=0,
                retrieval_time_ms=0.0,
                generation_time_ms=0.0,
            ),
        )
