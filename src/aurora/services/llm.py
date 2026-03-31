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
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}") + 1
        if start != -1 and end > start:
            return json.loads(content[start:end])
        raise


class LLMService:
    def __init__(self, api_key: str, base_url: str, model: str) -> None:
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        self._model = model

    async def generate_answer(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        profile: UserProfile | None = None,
    ) -> AskResponse:
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
                retrieval_time_ms=0.0,
                generation_time_ms=0.0,
            ),
        )

    @staticmethod
    def _no_data_response() -> AskResponse:
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
