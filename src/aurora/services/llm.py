"""LLM answer generation — Gemini 2.0 Flash via OpenRouter (see README for benchmarks)."""

import json
from typing import Any

from loguru import logger
from openai import AsyncOpenAI

from aurora.models import AskMetadata, AskResponse, JudgeResult, RetrievedChunk, UserProfile

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
    """Build prompt: profile (always injected) + retrieved chunks + question."""
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


def parse_llm_response(content: str) -> dict[str, Any]:
    """Parse JSON from LLM response, with fallback for text-wrapped JSON."""
    try:
        result: dict[str, Any] = json.loads(content)
        return result
    except json.JSONDecodeError:
        # Fallback: extract JSON object from surrounding text
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            result = json.loads(content[start:end])
            return result
        raise


class LLMService:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        # OpenAI SDK pointed at OpenRouter — gives us JSON mode + async for free
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def generate_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        profile: UserProfile | None = None,
    ) -> AskResponse:
        # No relevant chunks → skip LLM call, return no-data response
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
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        content = response.choices[0].message.content or "{}"
        parsed = parse_llm_response(content)

        return AskResponse(
            answer=parsed.get("answer", "Unable to generate an answer."),
            confidence=min(max(float(parsed.get("confidence", 0.0)), 0.0), 1.0),
            sources=parsed.get("sources", []),
            metadata=AskMetadata(
                reasoning=parsed.get("reasoning", ""),
                sources_considered=len(chunks),
                retrieval_time_ms=0.0,  # Filled by router
                generation_time_ms=0.0,
            ),
        )

    # --- Judge (opt-in via judge=true in request) ---
    # Used during development to validate answer quality against our hybrid confidence
    # scoring. Not part of the default request flow — adds ~1s latency when enabled.
    # See README "LLM-as-Judge Verification" for results.

    async def judge_answer(
        self,
        question: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> JudgeResult:
        """Independent LLM judge — evaluates if the answer is grounded in the sources.

        Uses the same model but a different role: instead of answering the question,
        it evaluates whether someone else's answer is supported by the evidence.
        """
        sources_text = "\n".join(f"[{c.source_id}] {c.text}" for c in chunks)

        judge_prompt = (
            f"You are an independent judge evaluating answer quality.\n\n"
            f"QUESTION: {question}\n\n"
            f"ANSWER GIVEN: {answer}\n\n"
            f"SOURCE DATA:\n{sources_text}\n\n"
            f"Evaluate: Is the answer factually supported by the source data?\n"
            f"Respond with JSON only:\n"
            f'{{"score": 0.0-1.0, "assessment": "brief explanation", "agrees_with_answer": true/false}}\n'
            f"Score guide: 1.0 = fully supported, 0.7 = mostly supported, "
            f"0.5 = partially, 0.3 = weakly, 0.0 = unsupported or hallucinated."
        )

        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            content = response.choices[0].message.content or "{}"
            parsed = parse_llm_response(content)
            return JudgeResult(
                score=min(max(float(parsed.get("score", 0.0)), 0.0), 1.0),
                assessment=parsed.get("assessment", ""),
                agrees_with_answer=parsed.get("agrees_with_answer", False),
            )
        except Exception as e:
            logger.error(f"Judge call failed: {e}")
            return JudgeResult(score=0.0, assessment=f"Judge error: {e}", agrees_with_answer=False)

    @staticmethod
    def _no_data_response() -> AskResponse:
        """Graceful no-data response — skips LLM call entirely."""
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
