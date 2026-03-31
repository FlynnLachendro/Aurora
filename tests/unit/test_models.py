import pytest
from pydantic import ValidationError

from aurora.models import (
    AskMetadata,
    AskRequest,
    AskResponse,
    CalendarEvent,
    Message,
    PaginatedResponse,
    SpotifyStream,
    UserProfile,
    WhoopRecord,
)


class TestMessage:
    def test_parse_from_api_json(self):
        data = {
            "id": "b1e9bb83-18be-4b90-bbb8-83b7428e8e21",
            "user_id": "cd3a350e-dbd2-408f-afa0-16a072f56d23",
            "user_name": "Sophia Al-Farsi",
            "timestamp": "2025-05-05T07:47:20.159073+00:00",
            "message": "Please book a private jet to Paris for this Friday.",
        }
        msg = Message(**data)
        assert msg.id == "b1e9bb83-18be-4b90-bbb8-83b7428e8e21"
        assert msg.user_name == "Sophia Al-Farsi"
        assert "Paris" in msg.message


class TestCalendarEvent:
    def test_parse_from_api_json(self):
        data = {
            "id": "evt_0001",
            "title": "Vela Daily Standup",
            "start": "2026-02-13T09:00:00",
            "end": "2026-02-13T10:00:00",
            "type": "meeting",
            "location": "Zoom",
            "attendees": ["Maya Chen", "Jake Torres"],
            "notes": "",
            "recurring": True,
            "all_day": False,
        }
        event = CalendarEvent(**data)
        assert event.title == "Vela Daily Standup"
        assert event.recurring is True
        assert len(event.attendees) == 2


class TestSpotifyStream:
    def test_parse_from_api_json(self):
        data = {
            "stream_id": "sp_00002",
            "date": "2026-02-13",
            "timestamp": "2026-02-13T08:03:00",
            "type": "music",
            "title": "Feel It Still — Portugal. The Man",
            "artist_or_show": "Commute 🚗",
            "duration_ms": 180000,
            "context": "commute",
        }
        stream = SpotifyStream(**data)
        assert stream.stream_id == "sp_00002"
        assert stream.duration_ms == 180000


class TestWhoopRecord:
    def test_parse_from_api_json(self):
        data = {
            "date": "2026-02-13",
            "recovery": {
                "score": 53,
                "hrv_ms": 61.7,
                "rhr_bpm": 52.4,
                "skin_temp_celsius": 36.2,
            },
            "sleep": {
                "bedtime": "2026-02-13T00:30:00",
                "wake_time": "2026-02-13T07:55:48",
                "duration_hours": 7.43,
                "quality_score": 60,
                "stages": {"rem_hours": 1.6, "deep_hours": 1.14, "light_hours": 4.7},
                "disruptions": 0,
                "respiratory_rate": 17.4,
                "spo2_avg": 98.0,
            },
            "strain": {
                "score": 6.6,
                "calories_burned": 232,
                "steps": 5801,
                "activities": [],
                "active_hours": 0.0,
            },
        }
        record = WhoopRecord(**data)
        assert record.recovery.score == 53
        assert record.sleep.duration_hours == 7.43
        assert record.strain.steps == 5801


class TestUserProfile:
    def test_parse_from_api_json(self):
        data = {
            "name": "James Fletcher",
            "date_of_birth": "1990-08-12",
            "summary": "James Fletcher is a founder deep in a fundraising sprint.",
        }
        profile = UserProfile(**data)
        assert profile.name == "James Fletcher"


class TestPaginatedResponse:
    def test_parse_messages(self):
        data = {
            "total": 3349,
            "items": [
                {
                    "id": "msg-1",
                    "user_id": "u-1",
                    "user_name": "Test",
                    "timestamp": "2025-01-01T00:00:00+00:00",
                    "message": "Hello",
                }
            ],
        }
        resp = PaginatedResponse[dict](**data)
        assert resp.total == 3349
        assert len(resp.items) == 1


class TestAskRequest:
    def test_valid_question(self):
        req = AskRequest(question="What is Sophia's favorite restaurant?")
        assert req.question == "What is Sophia's favorite restaurant?"

    def test_empty_question_rejected(self):
        with pytest.raises(ValidationError):
            AskRequest(question="")

    def test_too_long_question_rejected(self):
        with pytest.raises(ValidationError):
            AskRequest(question="x" * 1001)


class TestAskResponse:
    def test_valid_response(self):
        resp = AskResponse(
            answer="Le Cinq in Paris.",
            confidence=0.85,
            sources=["msg-003"],
            metadata=AskMetadata(
                reasoning="Found explicit mention.",
                sources_considered=5,
                retrieval_time_ms=12.5,
                generation_time_ms=450.0,
            ),
        )
        assert resp.confidence == 0.85

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            AskResponse(
                answer="Test",
                confidence=1.5,
                sources=[],
                metadata=AskMetadata(
                    reasoning="",
                    sources_considered=0,
                    retrieval_time_ms=0,
                    generation_time_ms=0,
                ),
            )
