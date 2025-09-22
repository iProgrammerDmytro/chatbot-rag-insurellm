from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.core.config import get_settings

settings = get_settings()


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    k: int = Field(default=8, ge=1, le=10, description="Top-K passages to retrieve")
    ns: Optional[str] = Field(None, description="Namespace filter")

    db: str = Field(default=settings.chroma_dir, description="Chroma directory")
    collection: str = Field(
        default=settings.chroma_collection,
        description="Chroma collection",
    )
    model: str = Field(
        default=settings.embedding_model,
        description="Embedding model id",
    )

    llm_model: Optional[str] = Field(
        default=settings.openai_model, description="OpenAI chat model"
    )
    include_passages: bool = Field(
        default=True,
        description="Include retrieved passages in response for debugging",
    )


class Source(BaseModel):
    title: Optional[str] = None
    source: Optional[str] = None
    distance: Optional[float] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[Source] = []
    used: Dict[str, Any] = {}
    passages: Optional[List[str]] = None  # only present if include_passages=True
