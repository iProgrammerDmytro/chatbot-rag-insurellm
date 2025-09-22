from __future__ import annotations

from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

settings = get_settings()


class Retriever:
    """
    Callable retriever:
      - vector() returns batch
      - __call__() returns (documents, metadatas, distances) from Chroma
      - prints are handled by caller/cli to keep class single-responsibility
    """

    def __init__(
        self,
        db_path: Optional[str] = settings.chroma_dir,
        collection: Optional[str] = settings.chroma_collection,
        model_name: str = settings.embedding_model,
    ) -> None:
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name=collection)
        self.model = SentenceTransformer(model_name)

    # return a batch (list-of-vectors)
    def vector(self, text: str):
        return self.model.encode([text])

    def __call__(self, question: str, k: int = 5, namespace: Optional[str] = None):
        where = {"namespace": namespace} if namespace else None

        q = self.vector(question)  # shape (1, dim)
        results = self.collection.query(
            query_embeddings=q.tolist(),
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        documents = results["documents"][0][:] if results.get("documents") else []
        metadatas = results["metadatas"][0][:] if results.get("metadatas") else []
        distances = results["distances"][0][:] if results.get("distances") else []

        return documents, metadatas, distances
