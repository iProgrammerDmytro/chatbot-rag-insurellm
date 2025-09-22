from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    api_prefix: str = "/api"
    project_name: str = "Insurellm QA Chatbot"

    # OpenAI
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"

    # Chroma
    kb_root: str = "knowledge-base"
    chroma_dir: str = "chroma-db"
    chroma_collection: str = "knowledge-base"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Startup
    auto_ingest: bool = False
    drop_on_startup: bool = False

    class Config:
        env_file = ".env"


@lru_cache
def get_settings() -> Settings:
    return Settings()
