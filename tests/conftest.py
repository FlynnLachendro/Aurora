import tempfile

import pytest

from aurora.config import Settings
from aurora.models import (
    CalendarEvent,
    Document,
    Message,
    RetrievedChunk,
    SpotifyStream,
    UserProfile,
    WhoopActivity,
    WhoopRecord,
    WhoopRecovery,
    WhoopSleep,
    WhoopSleepStages,
    WhoopStrain,
)
from aurora.services.embeddings import VectorStore


@pytest.fixture
def sample_messages() -> list[Message]:
    return [
        Message(
            id="msg-001",
            user_id="user-001",
            user_name="Sophia Al-Farsi",
            timestamp="2025-05-05T07:47:20.159073+00:00",
            message="Please book a private jet to Paris for this Friday.",
        ),
        Message(
            id="msg-002",
            user_id="user-002",
            user_name="Fatima El-Tahir",
            timestamp="2024-11-14T20:03:44.159235+00:00",
            message="Can you confirm my dinner reservation at The French Laundry for four people tonight?",
        ),
        Message(
            id="msg-003",
            user_id="user-001",
            user_name="Sophia Al-Farsi",
            timestamp="2025-06-10T10:00:00+00:00",
            message="I loved dining at Le Cinq in Paris last month. Please book it again for next week.",
        ),
        Message(
            id="msg-004",
            user_id="user-003",
            user_name="Armand Dupont",
            timestamp="2025-03-09T02:25:23+00:00",
            message="I need two tickets to the opera in Milan this Saturday.",
        ),
        Message(
            id="msg-005",
            user_id="user-001",
            user_name="Sophia Al-Farsi",
            timestamp="2025-07-01T12:00:00+00:00",
            message="Please remember I prefer aisle seats during my flights.",
        ),
    ]


@pytest.fixture
def sample_calendar_events() -> list[CalendarEvent]:
    return [
        CalendarEvent(
            id="evt_0001",
            title="Vela Daily Standup",
            start="2026-02-13T09:00:00",
            end="2026-02-13T10:00:00",
            type="meeting",
            location="Zoom",
            attendees=["Maya Chen", "Jake Torres"],
            notes="",
            recurring=True,
            all_day=False,
        ),
        CalendarEvent(
            id="evt_0058",
            title="Deep Work — Strategy & Writing",
            start="2026-02-13T08:00:00",
            end="2026-02-13T12:00:00",
            type="blocked",
            location="San Francisco, CA",
            attendees=[],
            notes="No meetings. Thinking time.",
            recurring=False,
            all_day=False,
        ),
    ]


@pytest.fixture
def sample_spotify_streams() -> list[SpotifyStream]:
    return [
        SpotifyStream(
            stream_id="sp_00002",
            date="2026-02-13",
            timestamp="2026-02-13T08:03:00",
            type="music",
            title="Feel It Still",
            artist_or_show="Portugal. The Man",
            duration_ms=180000,
            context="commute",
        ),
        SpotifyStream(
            stream_id="sp_00005",
            date="2026-02-13",
            timestamp="2026-02-13T09:00:00",
            type="music",
            title="Weightless",
            artist_or_show="Marconi Union",
            duration_ms=480000,
            context="deep_work",
        ),
    ]


@pytest.fixture
def sample_whoop_records() -> list[WhoopRecord]:
    return [
        WhoopRecord(
            date="2026-02-15",
            recovery=WhoopRecovery(score=67, hrv_ms=73.7, rhr_bpm=48.9, skin_temp_celsius=36.6),
            sleep=WhoopSleep(
                bedtime="2026-02-15T22:56:00",
                wake_time="2026-02-16T06:57:48",
                duration_hours=8.03,
                quality_score=90,
                stages=WhoopSleepStages(rem_hours=1.81, deep_hours=0.99, light_hours=5.23),
                disruptions=0,
                respiratory_rate=14.8,
                spo2_avg=97.8,
            ),
            strain=WhoopStrain(
                score=21.0,
                calories_burned=798,
                steps=5973,
                activities=[WhoopActivity(type="Tennis", duration_min=101, calories=559, avg_hr=138, max_hr=170)],
                active_hours=1.68,
            ),
        ),
    ]


@pytest.fixture
def sample_profile() -> UserProfile:
    return UserProfile(
        name="James Fletcher",
        date_of_birth="1990-08-12",
        summary="James Fletcher is a founder deep in a fundraising sprint.",
    )


@pytest.fixture
def sample_documents(sample_messages: list[Message]) -> list[Document]:
    from aurora.services.ingestion import message_to_document

    return [message_to_document(msg) for msg in sample_messages]


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            source_id="msg-001",
            source_type="message",
            text="[2025-05-05] Sophia Al-Farsi: Please book a private jet to Paris for this Friday.",
            distance=0.3,
            user_name="Sophia Al-Farsi",
            timestamp="2025-05-05T07:47:20.159073+00:00",
        ),
        RetrievedChunk(
            source_id="msg-003",
            source_type="message",
            text="[2025-06-10] Sophia Al-Farsi: I loved dining at Le Cinq in Paris last month.",
            distance=0.5,
            user_name="Sophia Al-Farsi",
            timestamp="2025-06-10T10:00:00+00:00",
        ),
    ]


@pytest.fixture
def tmp_chroma_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def vector_store(tmp_chroma_dir: str) -> VectorStore:
    return VectorStore(persist_dir=tmp_chroma_dir, embedding_model="text-embedding-3-small")


@pytest.fixture
def test_settings(tmp_chroma_dir: str) -> Settings:
    return Settings(
        openrouter_api_key="test-key",
        chroma_persist_dir=tmp_chroma_dir,
    )
