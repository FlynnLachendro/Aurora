import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from aurora.models import AskResponse, RetrievedChunk, UserProfile
from aurora.services.llm import LLMService, build_user_prompt, parse_llm_response


class TestBuildUserPrompt:
    def test_includes_question(self, sample_chunks: list[RetrievedChunk]):
        prompt = build_user_prompt("What is Sophia's favorite city?", sample_chunks, None)
        assert "What is Sophia's favorite city?" in prompt

    def test_includes_chunks(self, sample_chunks: list[RetrievedChunk]):
        prompt = build_user_prompt("test question", sample_chunks, None)
        assert "msg-001" in prompt
        assert "msg-003" in prompt
        assert "private jet" in prompt

    def test_includes_source_type(self, sample_chunks: list[RetrievedChunk]):
        prompt = build_user_prompt("test", sample_chunks, None)
        assert "[Type: message]" in prompt

    def test_includes_profile_when_provided(self, sample_chunks: list[RetrievedChunk], sample_profile: UserProfile):
        prompt = build_user_prompt("test", sample_chunks, sample_profile)
        assert "James Fletcher" in prompt
        assert "1990-08-12" in prompt

    def test_no_profile_section_when_none(self, sample_chunks: list[RetrievedChunk]):
        prompt = build_user_prompt("test", sample_chunks, None)
        assert "PRIMARY USER PROFILE" not in prompt


class TestParseLlmResponse:
    def test_valid_json(self):
        raw = json.dumps({"answer": "test", "confidence": 0.9, "sources": [], "reasoning": "ok"})
        result = parse_llm_response(raw)
        assert result["answer"] == "test"

    def test_json_with_surrounding_text(self):
        raw = 'Here is the answer: {"answer": "test", "confidence": 0.5, "sources": [], "reasoning": "ok"} done.'
        result = parse_llm_response(raw)
        assert result["answer"] == "test"

    def test_invalid_json_raises(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            parse_llm_response("not json at all")


class TestLLMService:
    async def test_generate_answer_with_chunks(self, sample_chunks, sample_profile):
        service = LLMService(api_key="test", base_url="https://test.com/v1", model="test-model")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "answer": "Le Cinq in Paris",
                            "confidence": 0.85,
                            "sources": ["msg-003"],
                            "reasoning": "Found explicit mention in msg-003.",
                        }
                    )
                )
            )
        ]

        service._client = AsyncMock()
        service._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await service.generate_answer("favorite restaurant?", sample_chunks, sample_profile)
        assert isinstance(result, AskResponse)
        assert result.answer == "Le Cinq in Paris"
        assert result.confidence == 0.85
        assert "msg-003" in result.sources

    async def test_no_data_response_with_empty_chunks(self, sample_profile):
        service = LLMService(api_key="test", base_url="https://test.com/v1", model="test-model")
        result = await service.generate_answer("anything?", [], sample_profile)
        assert result.confidence == 0.0
        assert result.sources == []
        assert "don't have enough" in result.answer

    async def test_handles_malformed_llm_output(self, sample_chunks, sample_profile):
        service = LLMService(api_key="test", base_url="https://test.com/v1", model="test-model")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='Sure! {"answer": "test answer", "confidence": 0.7, "sources": [], "reasoning": "found it"} hope that helps!'
                )
            )
        ]

        service._client = AsyncMock()
        service._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await service.generate_answer("test?", sample_chunks, sample_profile)
        assert result.answer == "test answer"

    async def test_confidence_clamped(self, sample_chunks, sample_profile):
        service = LLMService(api_key="test", base_url="https://test.com/v1", model="test-model")

        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        {
                            "answer": "test",
                            "confidence": 5.0,
                            "sources": [],
                            "reasoning": "very confident",
                        }
                    )
                )
            )
        ]

        service._client = AsyncMock()
        service._client.chat.completions.create = AsyncMock(return_value=mock_response)

        result = await service.generate_answer("test?", sample_chunks, sample_profile)
        assert result.confidence == 1.0
