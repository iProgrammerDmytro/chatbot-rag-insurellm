from __future__ import annotations

import argparse

from app.core.config import get_settings
from app.vector_db.loader import Loader

settings = get_settings()


def cmd_load(args):
    loader = Loader(
        kb_root=args.kb,
        db_path=args.db,
        collection=args.collection,
        model_name=args.model,
    )
    n = loader(namespace=args.ns, drop=args.drop)
    print(f"Ingested {n} files into '{args.collection}' at '{args.db}'.")


def _add_load_args(sp: argparse.ArgumentParser):
    sp.add_argument("--kb", default=settings.kb_root)
    sp.add_argument("--db", default=settings.chroma_dir)
    sp.add_argument("--collection", default=settings.chroma_collection)
    sp.add_argument("--model", default=settings.embedding_model)
    sp.add_argument(
        "--ns",
        default=None,
        help="namespace (company|contracts|employees|products|root)",
    )
    sp.add_argument(
        "--drop", action="store_true", help="delete+recreate collection before ingest"
    )
    sp.set_defaults(func=cmd_load)


def main():
    p = argparse.ArgumentParser(description="RAG over Markdown (Chroma)")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_load = sub.add_parser("load", help="(re)ingest Markdown into Chroma")
    _add_load_args(p_load)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
