"""Fetches all 5 Aurora API sources concurrently, converts records to natural language docs."""

import asyncio
from typing import Any

import httpx
from loguru import logger

from aurora.config import Settings
from aurora.core.constants import (
    API_CALENDAR_PATH,
    API_MESSAGES_PATH,
    API_PROFILE_PATH,
    API_SPOTIFY_PATH,
    API_WHOOP_PATH,
    SOURCE_TYPE_CALENDAR,
    SOURCE_TYPE_MESSAGE,
    SOURCE_TYPE_PROFILE,
    SOURCE_TYPE_SPOTIFY,
    SOURCE_TYPE_WHOOP,
)
from aurora.models import (
    CalendarEvent,
    Document,
    Message,
    SpotifyStream,
    UserProfile,
    WhoopRecord,
)


async def _get_with_retry(
    client: httpx.AsyncClient,
    path: str,
    params: dict[str, Any],
    max_retries: int = 3,
) -> httpx.Response:
    """GET with retry for Aurora's intermittent 4xx/5xx errors."""
    for attempt in range(max_retries):
        resp = await client.get(path, params=params)
        if resp.status_code < 400:
            return resp
        if attempt < max_retries - 1:
            wait = 1.0 * (attempt + 1)
            logger.warning(f"Retry {attempt + 1}/{max_retries} for {path} (status {resp.status_code})")
            await asyncio.sleep(wait)
    resp.raise_for_status()
    return resp


async def fetch_all_paginated(
    client: httpx.AsyncClient,
    path: str,
    page_size: int = 100,
) -> list[dict[str, Any]]:
    """Paginate through an Aurora API endpoint, collecting all items."""
    items: list[dict[str, Any]] = []
    skip = 0

    first_resp = await _get_with_retry(client, path, {"skip": 0, "limit": page_size})
    data = first_resp.json()
    total = data["total"]
    items.extend(data["items"])
    skip += page_size

    while skip < total:
        resp = await _get_with_retry(client, path, {"skip": skip, "limit": page_size})
        items.extend(resp.json()["items"])
        skip += page_size

    logger.info(f"Fetched {len(items)}/{total} items from {path}")
    return items


async def fetch_profile(client: httpx.AsyncClient) -> dict[str, Any]:
    resp = await client.get(API_PROFILE_PATH)
    resp.raise_for_status()
    result: dict[str, Any] = resp.json()
    return result


# --- Document text constructors ---
# Convert API records to natural language for embedding (JSON embeds poorly).


def message_to_document(msg: Message) -> Document:
    date = msg.timestamp[:10] if msg.timestamp else ""
    text = f"[{date}] {msg.user_name}: {msg.message}"
    return Document(
        source_id=msg.id,
        source_type=SOURCE_TYPE_MESSAGE,
        text=text,
        user_name=msg.user_name,
        timestamp=msg.timestamp,
    )


def calendar_to_document(event: CalendarEvent) -> Document:
    parts = [f"Calendar event: {event.title} on {event.start} to {event.end}"]
    if event.location:
        parts.append(f"Location: {event.location}")
    parts.append(f"Type: {event.type}")
    if event.attendees:
        parts.append(f"Attendees: {', '.join(event.attendees)}")
    if event.notes:
        parts.append(f"Notes: {event.notes}")
    if event.recurring:
        parts.append("Recurring: yes")
    if event.all_day:
        parts.append("All day event")
    text = ". ".join(parts) + "."
    return Document(
        source_id=event.id,
        source_type=SOURCE_TYPE_CALENDAR,
        text=text,
        timestamp=event.start,
    )


def spotify_to_document(stream: SpotifyStream) -> Document:
    duration_min = round(stream.duration_ms / 60000, 1)
    text = (
        f'Spotify {stream.type}: "{stream.title}" by/from {stream.artist_or_show} '
        f"on {stream.date} at {stream.timestamp[11:16] if len(stream.timestamp) > 11 else stream.date}. "
        f"Context: {stream.context}. Duration: {duration_min} min."
    )
    return Document(
        source_id=stream.stream_id,
        source_type=SOURCE_TYPE_SPOTIFY,
        text=text,
        timestamp=stream.timestamp,
    )


def whoop_to_document(record: WhoopRecord) -> Document:
    activities_str = ""
    if record.strain.activities:
        activity_names = [a.type for a in record.strain.activities]
        activities_str = f" Activities: {', '.join(activity_names)}."

    text = (
        f"Health data for {record.date}: "
        f"Recovery score {record.recovery.score}/100, "
        f"HRV {record.recovery.hrv_ms}ms, "
        f"RHR {record.recovery.rhr_bpm}bpm. "
        f"Sleep: {record.sleep.duration_hours}h, "
        f"quality {record.sleep.quality_score}/100, "
        f"bedtime {record.sleep.bedtime[11:16] if len(record.sleep.bedtime) > 11 else record.sleep.bedtime}, "
        f"wake {record.sleep.wake_time[11:16] if len(record.sleep.wake_time) > 11 else record.sleep.wake_time}. "
        f"Strain: {record.strain.score}, "
        f"{record.strain.calories_burned} cal, "
        f"{record.strain.steps} steps.{activities_str}"
    )
    return Document(
        source_id=f"whoop_{record.date}",
        source_type=SOURCE_TYPE_WHOOP,
        text=text,
        timestamp=record.date,
    )


def profile_to_document(profile: UserProfile) -> Document:
    text = f"User profile: {profile.name}, born {profile.date_of_birth}. {profile.summary}"
    return Document(
        source_id=f"profile_{profile.name.lower().replace(' ', '_')}",
        source_type=SOURCE_TYPE_PROFILE,
        text=text,
        user_name=profile.name,
    )


async def fetch_all_data(
    settings: Settings,
) -> tuple[list[Document], UserProfile]:
    """Fetch all 5 endpoints concurrently, convert to documents. Needs follow_redirects."""
    async with httpx.AsyncClient(base_url=settings.aurora_api_base_url, timeout=30.0, follow_redirects=True) as client:
        messages_raw, calendar_raw, spotify_raw, whoop_raw, profile_raw = await asyncio.gather(
            fetch_all_paginated(client, API_MESSAGES_PATH, settings.aurora_api_page_size),
            fetch_all_paginated(client, API_CALENDAR_PATH, settings.aurora_api_page_size),
            fetch_all_paginated(client, API_SPOTIFY_PATH, settings.aurora_api_page_size),
            fetch_all_paginated(client, API_WHOOP_PATH, settings.aurora_api_page_size),
            fetch_profile(client),
        )

    documents: list[Document] = []

    for raw in messages_raw:
        documents.append(message_to_document(Message(**raw)))
    for raw in calendar_raw:
        documents.append(calendar_to_document(CalendarEvent(**raw)))
    for raw in spotify_raw:
        documents.append(spotify_to_document(SpotifyStream(**raw)))
    for raw in whoop_raw:
        documents.append(whoop_to_document(WhoopRecord(**raw)))

    profile = UserProfile(**profile_raw)
    documents.append(profile_to_document(profile))

    logger.info(f"Total documents prepared: {len(documents)}")
    return documents, profile
