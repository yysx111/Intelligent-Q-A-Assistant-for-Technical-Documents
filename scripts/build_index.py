from __future__ import annotations

import argparse
import pathlib

from app.rag.document_loader import load_document_from_bytes
from app.rag.text_splitter import split_documents
from app.rag.vector_store import get_vector_store


def iter_files(root: pathlib.Path) -> list[pathlib.Path]:
    exts = {".pdf", ".txt", ".md", ".html", ".htm"}
    return [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw", help="输入目录")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    args = parser.parse_args()

    root = pathlib.Path(args.input)
    files = iter_files(root) if root.exists() else []
    if not files:
        print(f"未找到可索引文件: {root}")
        return

    vector_store = get_vector_store()
    total_chunks = 0
    for p in files:
        data = p.read_bytes()
        docs = load_document_from_bytes(p.name, data, source=str(p))
        chunks = split_documents(docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        if chunks:
            vector_store.add_documents(chunks)
            total_chunks += len(chunks)
            print(f"已索引: {p} -> {len(chunks)}")
    print(f"完成，总片段数: {total_chunks}")


if __name__ == "__main__":
    main()
