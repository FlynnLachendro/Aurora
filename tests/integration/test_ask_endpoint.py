import json
import tempfile
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from aurora.models import Document, UserProfile
from aurora.routers.ask import router
from aurora.services.embeddings import VectorStore
from aurora.services.llm import LLMService
from aurora.services.retrieval import RetrievalService


def make_mock_llm_response(answer: str, confidence: float, sources: list[str], reasoning: str):
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "answer": answer,
                        "confidence": confidence,
                        "sources": sources,
                        "reasoning": reasoning,
                    }
                )
            )
        )
    ]
    return mock_response


@pytest.fixture
def integration_documents() -> list[Document]:
    return [
        Document(
            source_id="msg-001",
            source_type="message",
            text="[2025-05-05] Sophia Al-Farsi: Please book a private jet to Paris for this Friday.",
            user_name="Sophia Al-Farsi",
            timestamp="2025-05-05T07:47:20+00:00",
        ),
        Document(
            source_id="msg-002",
            source_type="message",
            text="[2024-11-14] Fatima El-Tahir: Can you confirm my dinner reservation at The French Laundry for four people tonight?",
            user_name="Fatima El-Tahir",
            timestamp="2024-11-14T20:03:44+00:00",
        ),
        Document(
            source_id="msg-003",
            source_type="message",
            text="[2025-06-10] Sophia Al-Farsi: I loved dining at Le Cinq in Paris last month. Please book it again for next week.",
            user_name="Sophia Al-Farsi",
            timestamp="2025-06-10T10:00:00+00:00",
        ),
        Document(
            source_id="evt_0001",
            source_type="calendar",
            text="Calendar event: Vela Daily Standup on 2026-02-13T09:00:00 to 2026-02-13T10:00:00. Location: Zoom. Type: meeting. Attendees: Maya Chen, Jake Torres. Recurring: yes.",
            timestamp="2026-02-13T09:00:00",
        ),
        Document(
            source_id="sp_00005",
            source_type="spotify",
            text='Spotify music: "Weightless" by/from Marconi Union on 2026-02-13 at 09:00. Context: deep_work. Duration: 8.0 min.',
            timestamp="2026-02-13T09:00:00",
        ),
        Document(
            source_id="whoop_2026-02-15",
            source_type="whoop",
            text="Health data for 2026-02-15: Recovery score 67/100, HRV 73.7ms, RHR 48.9bpm. Sleep: 8.03h, quality 90/100, bedtime 22:56, wake 06:57. Strain: 21.0, 798 cal, 5973 steps. Activities: Tennis.",
            timestamp="2026-02-15",
        ),
    ]


@pytest.fixture
def integration_app(integration_documents: list[Document]):
    with tempfile.TemporaryDirectory() as tmpdir:
        vector_store = VectorStore(persist_dir=tmpdir, embedding_model="text-embedding-3-small")
        vector_store.ingest(integration_documents)

        retrieval_service = RetrievalService(vector_store=vector_store, top_k=10, similarity_threshold=1.5)

        llm_service = LLMService(api_key="test-key", base_url="https://test.com/v1", model="test-model")
        llm_service._client = AsyncMock()

        profile = UserProfile(
            name="James Fletcher",
            date_of_birth="1990-08-12",
            summary="James Fletcher is a founder deep in a fundraising sprint.",
        )

        app = FastAPI()
        app.include_router(router)
        app.state.retrieval_service = retrieval_service
        app.state.llm_service = llm_service
        app.state.profile = profile

        yield app, llm_service


class TestAskEndpoint:
    async def test_factual_question(self, integration_app):
        app, llm_service = integration_app
        llm_service._client.chat.completions.create = AsyncMock(
            return_value=make_mock_llm_response(
                "Sophia Al-Farsi requested a private jet to Paris.",
                0.95,
                ["msg-001"],
                "Found direct request in msg-001.",
            )
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ask", json={"question": "What did Sophia request?"})

        assert resp.status_code == 200
        data = resp.json()
        assert data["confidence"] > 0.0  # Hybrid score: retrieval distance + LLM confidence
        assert "msg-001" in data["sources"]

    async def test_calendar_question(self, integration_app):
        app, llm_service = integration_app
        llm_service._client.chat.completions.create = AsyncMock(
            return_value=make_mock_llm_response(
                "James has a Vela Daily Standup at 9am on Zoom.",
                0.9,
                ["evt_0001"],
                "Found recurring standup in evt_0001.",
            )
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ask", json={"question": "When is James's daily standup?"})

        assert resp.status_code == 200
        data = resp.json()
        assert "evt_0001" in data["sources"]

    async def test_health_question(self, integration_app):
        app, llm_service = integration_app
        llm_service._client.chat.completions.create = AsyncMock(
            return_value=make_mock_llm_response(
                "James played tennis on Feb 15 with a strain score of 21.",
                0.9,
                ["whoop_2026-02-15"],
                "Found activity data in whoop_2026-02-15.",
            )
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ask", json={"question": "What exercise did James do recently?"})

        assert resp.status_code == 200
        assert "whoop_2026-02-15" in resp.json()["sources"]

    async def test_music_question(self, integration_app):
        app, llm_service = integration_app
        llm_service._client.chat.completions.create = AsyncMock(
            return_value=make_mock_llm_response(
                "James listens to Weightless by Marconi Union during deep work.",
                0.85,
                ["sp_00005"],
                "Found deep_work context in sp_00005.",
            )
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ask", json={"question": "What does James listen to while working?"})

        assert resp.status_code == 200
        assert "sp_00005" in resp.json()["sources"]

    async def test_empty_question_returns_422(self, integration_app):
        app, _ = integration_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ask", json={"question": ""})
        assert resp.status_code == 422

    async def test_missing_question_returns_422(self, integration_app):
        app, _ = integration_app
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ask", json={})
        assert resp.status_code == 422

    async def test_response_has_timing_metadata(self, integration_app):
        app, llm_service = integration_app
        llm_service._client.chat.completions.create = AsyncMock(
            return_value=make_mock_llm_response("Test answer.", 0.5, [], "No strong match.")
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ask", json={"question": "random question"})

        data = resp.json()
        assert "retrieval_time_ms" in data["metadata"]
        assert "generation_time_ms" in data["metadata"]
        assert data["metadata"]["retrieval_time_ms"] >= 0
        assert data["metadata"]["generation_time_ms"] >= 0

    async def test_response_structure(self, integration_app):
        app, llm_service = integration_app
        llm_service._client.chat.completions.create = AsyncMock(
            return_value=make_mock_llm_response("Answer.", 0.8, ["msg-001"], "Reasoning.")
        )

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            resp = await client.post("/ask", json={"question": "test"})

        data = resp.json()
        assert "answer" in data
        assert "confidence" in data
        assert "sources" in data
        assert "metadata" in data
        assert "reasoning" in data["metadata"]
