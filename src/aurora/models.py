"""Pydantic models — Aurora API types, POST /ask contract, internal Document/RetrievedChunk."""

from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T = TypeVar("T")


# --- Aurora API data models ---
# Match the JSON shapes from november7-730026606190.europe-west1.run.app


class PaginatedResponse(BaseModel, Generic[T]):
    total: int
    items: list[T]


class Message(BaseModel):
    id: str
    user_id: str
    user_name: str
    timestamp: str
    message: str


class CalendarEvent(BaseModel):
    id: str
    title: str
    start: str
    end: str
    type: str
    location: str
    attendees: list[str]
    notes: str
    recurring: bool
    all_day: bool


class SpotifyStream(BaseModel):
    stream_id: str
    date: str
    timestamp: str
    type: str
    title: str
    artist_or_show: str
    duration_ms: int
    context: str


class WhoopRecovery(BaseModel):
    score: float
    hrv_ms: float
    rhr_bpm: float
    skin_temp_celsius: float


class WhoopSleepStages(BaseModel):
    rem_hours: float
    deep_hours: float
    light_hours: float


class WhoopSleep(BaseModel):
    bedtime: str
    wake_time: str
    duration_hours: float
    quality_score: float
    stages: WhoopSleepStages
    disruptions: int
    respiratory_rate: float
    spo2_avg: float


class WhoopActivity(BaseModel):
    type: str
    duration_min: float
    calories: float
    avg_hr: float
    max_hr: float


class WhoopStrain(BaseModel):
    score: float
    calories_burned: float
    steps: int
    activities: list[WhoopActivity]
    active_hours: float


class WhoopRecord(BaseModel):
    date: str
    recovery: WhoopRecovery
    sleep: WhoopSleep
    strain: WhoopStrain


class UserProfile(BaseModel):
    name: str
    date_of_birth: str
    summary: str


# --- Request/Response models (POST /ask contract) ---


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    max_chunks: int | None = Field(None, ge=1, le=30, description="Override max chunks sent to LLM")
    judge: bool = Field(False, description="Run independent LLM judge to verify answer confidence")


class JudgeResult(BaseModel):
    """Opt-in judge (judge=true). Used to validate hybrid confidence during development."""

    score: float = Field(..., ge=0.0, le=1.0)
    assessment: str
    agrees_with_answer: bool


class AskMetadata(BaseModel):
    reasoning: str
    sources_considered: int
    retrieval_time_ms: float  # Time for embedding + ChromaDB search
    generation_time_ms: float  # Time for LLM call
    judge: JudgeResult | None = None  # Present when judge=true in request


class AskResponse(BaseModel):
    answer: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    sources: list[str]
    metadata: AskMetadata


# --- Internal models ---


class Document(BaseModel):
    """Prepared for ChromaDB — natural language text + metadata for filtering."""

    source_id: str
    source_type: str
    text: str
    user_name: str = ""
    timestamp: str = ""


class RetrievedChunk(BaseModel):
    """ChromaDB search result with cosine distance."""

    source_id: str
    source_type: str
    text: str
    distance: float
    user_name: str = ""
    timestamp: str = ""
