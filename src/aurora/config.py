from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openrouter_api_key: str
    openrouter_base_url: str = "https://openrouter.ai/api/v1"
    gemini_api_key: str = ""
    llm_model: str = "google/gemini-2.0-flash-001"
    embedding_model: str = "gemini-embedding-001"
    aurora_api_base_url: str = "https://november7-730026606190.europe-west1.run.app"
    aurora_api_page_size: int = 100
    chroma_persist_dir: str = "./chroma_data"
    retrieval_top_k: int = 15
    similarity_threshold: float = 1.2

    model_config = {"env_file": ".env", "extra": "ignore"}
