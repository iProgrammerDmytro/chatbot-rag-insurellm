from __future__ import annotations

from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.chatbot.router import router as chatbot_router
from app.core.config import get_settings
from app.core.bootstrap import ingest

settings = get_settings()


app = FastAPI(title=settings.project_name)

app.include_router(chatbot_router, prefix=settings.api_prefix)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.on_event("startup")
def startup_bootstrap() -> None:
    if settings.auto_ingest:
        print("Ingesting data...")
        ingest()
        print("Data ingested successfully")
