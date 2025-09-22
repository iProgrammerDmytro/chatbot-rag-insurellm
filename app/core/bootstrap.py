from app.vector_db.loader import Loader

from .config import get_settings

settings = get_settings()


def ingest() -> None:
    load = Loader()
    documents = load(drop=settings.drop_on_startup)
    print(f"Ingested {documents} documents")
