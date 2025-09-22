from __future__ import annotations

from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api.models.Collection import Collection
from sentence_transformers import SentenceTransformer

from app.core.config import get_settings

settings = get_settings()


class Loader:
    """
    Minimal loader:
    - scan Markdown
    - if drop=True: delete & create collection
    - add(ids, documents, embeddings, metadatas)
    """

    def __init__(
        self,
        kb_root: Optional[str | Path] = settings.kb_root,
        db_path: Optional[str] = settings.chroma_dir,
        collection: Optional[str] = settings.chroma_collection,
        model_name: str = settings.embedding_model,
    ) -> None:
        self.kb_root = Path(kb_root)
        self.db_path = db_path
        self.collection = collection
        self.model = SentenceTransformer(model_name)
        self.client = chromadb.PersistentClient(path=self.db_path)

    def __call__(self, namespace: Optional[str] = None, drop: bool = False) -> int:
        collection = self._get_collection(drop)

        ids, docs, metas = self._scan(namespace)
        if not ids:
            return 0

        vecs = self.model.encode(docs).tolist()
        collection.add(ids=ids, documents=docs, metadatas=metas, embeddings=vecs)
        return len(ids)

    def _get_collection(self, drop: bool) -> Collection:
        """Return collection, optionally dropping and recreating it."""
        if drop:
            try:
                self.client.delete_collection(self.collection)
            except Exception:
                pass

            return self.client.create_collection(self.collection)

        return self.client.get_or_create_collection(self.collection)

    def _rel_and_namespace(self, p: Path) -> tuple[str, str]:
        relative = p.relative_to(self.kb_root).as_posix()
        namespace = relative.split("/", 1)[0] if "/" in relative else "root"

        return relative, namespace

    def _files(self, namespace: Optional[str] = None):
        for p in self.kb_root.rglob("*.md"):
            if not p.is_file():
                continue

            if namespace:
                _, ns = self._rel_and_namespace(p)
                if ns != namespace:
                    continue

            yield p

    @staticmethod
    def _title(text: str, fallback: str) -> str:
        for line in text.splitlines():
            s = line.strip()
            if s.startswith("# "):
                return s[2:].strip()
            if s:
                return s

        return fallback

    def _scan(self, namespace: Optional[str] = None):
        ids, docs, metas = [], [], []

        for p in self._files(namespace):
            rel, ns = self._rel_and_namespace(p)
            text = p.read_text(encoding="utf-8", errors="ignore").strip()
            title = self._title(text, p.stem)

            ids.append(rel)  # stable id = relative path
            docs.append(text)  # full file content
            metas.append({"source": rel, "namespace": ns, "title": title})

        return ids, docs, metas
