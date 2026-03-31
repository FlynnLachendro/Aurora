import httpx
import pytest
import respx

from aurora.models import (
    CalendarEvent,
    Message,
    SpotifyStream,
    UserProfile,
)
from aurora.services.ingestion import (
    calendar_to_document,
    fetch_all_paginated,
    message_to_document,
    profile_to_document,
    spotify_to_document,
    whoop_to_document,
)


class TestFetchAllPaginated:
    @respx.mock
    async def test_single_page(self):
        respx.get("https://api.test/items", params={"skip": 0, "limit": 10}).mock(
            return_value=httpx.Response(
                200,
                json={"total": 3, "items": [{"id": 1}, {"id": 2}, {"id": 3}]},
            )
        )
        async with httpx.AsyncClient(base_url="https://api.test") as client:
            items = await fetch_all_paginated(client, "/items", page_size=10)
        assert len(items) == 3

    @respx.mock
    async def test_multiple_pages(self):
        respx.get("https://api.test/items", params={"skip": 0, "limit": 2}).mock(
            return_value=httpx.Response(
                200,
                json={"total": 5, "items": [{"id": 1}, {"id": 2}]},
            )
        )
        respx.get("https://api.test/items", params={"skip": 2, "limit": 2}).mock(
            return_value=httpx.Response(
                200,
                json={"total": 5, "items": [{"id": 3}, {"id": 4}]},
            )
        )
        respx.get("https://api.test/items", params={"skip": 4, "limit": 2}).mock(
            return_value=httpx.Response(
                200,
                json={"total": 5, "items": [{"id": 5}]},
            )
        )
        async with httpx.AsyncClient(base_url="https://api.test") as client:
            items = await fetch_all_paginated(client, "/items", page_size=2)
        assert len(items) == 5

    @respx.mock
    async def test_empty_response(self):
        respx.get("https://api.test/items", params={"skip": 0, "limit": 10}).mock(
            return_value=httpx.Response(
                200,
                json={"total": 0, "items": []},
            )
        )
        async with httpx.AsyncClient(base_url="https://api.test") as client:
            items = await fetch_all_paginated(client, "/items", page_size=10)
        assert len(items) == 0

    @respx.mock
    async def test_api_error_raises(self):
        respx.get("https://api.test/items", params={"skip": 0, "limit": 10}).mock(
            return_value=httpx.Response(500, text="Internal Server Error")
        )
        async with httpx.AsyncClient(base_url="https://api.test") as client:
            with pytest.raises(httpx.HTTPStatusError):
                await fetch_all_paginated(client, "/items", page_size=10)


class TestMessageToDocument:
    def test_basic_conversion(self, sample_messages: list[Message]):
        doc = message_to_document(sample_messages[0])
        assert doc.source_id == "msg-001"
        assert doc.source_type == "message"
        assert "Sophia Al-Farsi" in doc.text
        assert "private jet" in doc.text
        assert "[2025-05-05]" in doc.text

    def test_preserves_user_name(self, sample_messages: list[Message]):
        doc = message_to_document(sample_messages[0])
        assert doc.user_name == "Sophia Al-Farsi"


class TestCalendarToDocument:
    def test_basic_conversion(self, sample_calendar_events: list[CalendarEvent]):
        doc = calendar_to_document(sample_calendar_events[0])
        assert doc.source_id == "evt_0001"
        assert doc.source_type == "calendar"
        assert "Vela Daily Standup" in doc.text
        assert "Zoom" in doc.text
        assert "Maya Chen" in doc.text
        assert "Recurring: yes" in doc.text

    def test_event_with_notes(self, sample_calendar_events: list[CalendarEvent]):
        doc = calendar_to_document(sample_calendar_events[1])
        assert "Thinking time" in doc.text
        assert "Deep Work" in doc.text


class TestSpotifyToDocument:
    def test_basic_conversion(self, sample_spotify_streams: list[SpotifyStream]):
        doc = spotify_to_document(sample_spotify_streams[0])
        assert doc.source_id == "sp_00002"
        assert doc.source_type == "spotify"
        assert "Feel It Still" in doc.text
        assert "commute" in doc.text
        assert "3.0 min" in doc.text


class TestWhoopToDocument:
    def test_basic_conversion(self, sample_whoop_records):
        doc = whoop_to_document(sample_whoop_records[0])
        assert doc.source_id == "whoop_2026-02-15"
        assert doc.source_type == "whoop"
        assert "Recovery score 67" in doc.text
        assert "HRV 73.7ms" in doc.text
        assert "Tennis" in doc.text
        assert "8.03h" in doc.text


class TestProfileToDocument:
    def test_basic_conversion(self, sample_profile: UserProfile):
        doc = profile_to_document(sample_profile)
        assert doc.source_id == "profile_james_fletcher"
        assert doc.source_type == "profile"
        assert "James Fletcher" in doc.text
        assert "1990-08-12" in doc.text
        assert "fundraising" in doc.text
