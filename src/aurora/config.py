"""
Application settings loaded from environment variables via Pydantic Settings.

All tunables are env vars so they can be configured in Railway's dashboard
without code changes. The .env file is loaded for local development.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # LLM (Gemini Flash via OpenRouter)
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "google/gemini-2.0-flash-001"

    # Embedding (Gemini Embedding API via Google)
    gemini_api_key: str = ""
    embedding_model: str = "gemini-embedding-001"

    # Aurora's data API
    aurora_api_base_url: str = "https://november7-730026606190.europe-west1.run.app"
    aurora_api_page_size: int = 100

    # ChromaDB vector store
    chroma_persist_dir: str = "./chroma_data"

    # Retrieval tuning
    retrieval_top_k: int = 15  # Top-K results per query
    similarity_threshold: float = 1.2  # Cosine distance cutoff (higher = more lenient)

    model_config = {"env_file": ".env", "extra": "ignore"}
