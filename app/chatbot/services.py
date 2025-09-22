from __future__ import annotations

from typing import List, Tuple

from app.openai.llm import OpenAIChat
from app.vector_db.retriever import Retriever

from .exceptions import RetrievalFailed
from .schemas import AskRequest, AskResponse, Source



class ChatbotService:

    def retrieve(
        self, payload: AskRequest
    ) -> Tuple[List[str], List[dict], List[float]]:
        try:
            retrieve = Retriever(
                db_path=payload.db,
                collection=payload.collection,
                model_name=payload.model,
            )
            documents, metadatas, distances = retrieve(
                question=payload.question, k=payload.k, namespace=payload.ns
            )

            return documents, metadatas, distances
        except Exception as e:
            print(e)
            raise RetrievalFailed(type(e).__name__)

    async def get_answer(self, payload: AskRequest, documents: List[str]) -> str:
        chat = OpenAIChat(model=payload.llm_model)
        try:
            return await chat(question=payload.question, passages=documents)
        except Exception:
            return "I don't know."

    def build_sources(
        self,
        metadatas: List[dict] | None,
        distances: List[float] | None,
    ) -> List[Source]:
        sources: List[Source] = []

        if metadatas:
            for i, meta in enumerate(metadatas):
                sources.append(
                    Source(
                        title=(meta.get("title") if isinstance(meta, dict) else None),
                        source=(meta.get("source") if isinstance(meta, dict) else None),
                        distance=(
                            distances[i] if distances and i < len(distances) else None
                        ),
                    )
                )

        return sources

    def make_response(
        self,
        payload: AskRequest,
        answer: str,
        documents: List[str],
        sources: List[Source],
    ) -> AskResponse:
        return AskResponse(
            answer=answer,
            sources=sources,
            used={
                "db": payload.db,
                "collection": payload.collection,
                "embedding_model": payload.model,
                "llm_model": payload.llm_model,
                "k": payload.k,
                "ns": payload.ns,
            },
            passages=(documents if payload.include_passages else None),
        )

    async def ask(self, payload: AskRequest) -> AskResponse:
        documents, metadatas, distances = self.retrieve(payload)
        answer = await self.get_answer(payload, documents)
        sources = self.build_sources(metadatas, distances)

        return self.make_response(payload, answer, documents, sources)
