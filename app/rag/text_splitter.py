from __future__ import annotations

import uuid
from typing import List

from langchain.schema import Document


def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 120) -> list[str]:
    text = text or ""
    chunk_size = max(1, int(chunk_size))
    chunk_overlap = max(0, int(chunk_overlap))
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - chunk_overlap)
    return chunks


def split_documents(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> List[Document]:
    out: list[Document] = []
    for d in documents or []:
        base_meta = dict(d.metadata or {})
        doc_id = base_meta.get("doc_id") or uuid.uuid4().hex
        base_meta["doc_id"] = doc_id
        parts = split_text(d.page_content or "", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        for i, part in enumerate(parts):
            meta = dict(base_meta)
            meta["chunk_id"] = i
            out.append(Document(page_content=part, metadata=meta))
    return out
