from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import List, Tuple

import chromadb
from langchain.schema import Document

from app.config import get_settings


_WORD_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> list[str]:
    return [t.lower() for t in _WORD_RE.findall(text or "")]


@dataclass(frozen=True)
class _BM25Index:
    documents: list[Document]
    doc_tokens: list[list[str]]
    doc_len: list[int]
    avgdl: float
    df: dict[str, int]
    n_docs: int


class BM25Search:
    def __init__(self):
        self.settings = get_settings()
        self.client = chromadb.PersistentClient(path=self.settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="technical_docs",
            metadata={"hnsw:space": "cosine"},
        )
        self.k1 = 1.5
        self.b = 0.75
        self._index: _BM25Index | None = None
        self._indexed_count: int | None = None

    def _load_documents(self) -> list[Document]:
        data = self.collection.get(include=["documents", "metadatas", "ids"])
        docs = data.get("documents", []) or []
        metas = data.get("metadatas", []) or []
        ids = data.get("ids", []) or []
        out: list[Document] = []
        for i, content in enumerate(docs):
            meta = metas[i] if i < len(metas) else {}
            doc_id = ids[i] if i < len(ids) else None
            if isinstance(meta, dict) and doc_id and "doc_id" not in meta:
                meta = dict(meta)
                meta["doc_id"] = doc_id
            out.append(Document(page_content=content or "", metadata=meta or {}))
        return out

    def _ensure_index(self) -> None:
        count = int(self.collection.count())
        if self._index is not None and self._indexed_count == count:
            return
        documents = self._load_documents()
        tokens = [_tokenize(d.page_content) for d in documents]
        doc_len = [len(t) for t in tokens]
        n_docs = len(documents)
        avgdl = (sum(doc_len) / n_docs) if n_docs else 0.0
        df: dict[str, int] = {}
        for toks in tokens:
            for term in set(toks):
                df[term] = df.get(term, 0) + 1
        self._index = _BM25Index(
            documents=documents,
            doc_tokens=tokens,
            doc_len=doc_len,
            avgdl=avgdl,
            df=df,
            n_docs=n_docs,
        )
        self._indexed_count = count

    def _idf(self, term: str, df: int, n_docs: int) -> float:
        if n_docs <= 0:
            return 0.0
        return math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[Document, float]]:
        self._ensure_index()
        if not self._index or self._index.n_docs == 0:
            return []
        q_terms = _tokenize(query)
        if not q_terms:
            return []

        index = self._index
        scores = [0.0 for _ in range(index.n_docs)]
        q_terms_unique = set(q_terms)

        for term in q_terms_unique:
            df = index.df.get(term, 0)
            idf = self._idf(term, df, index.n_docs)
            if idf <= 0:
                continue
            for i, toks in enumerate(index.doc_tokens):
                tf = toks.count(term)
                if tf == 0:
                    continue
                dl = index.doc_len[i]
                denom = tf + self.k1 * (1 - self.b + self.b * (dl / (index.avgdl or 1.0)))
                score = idf * (tf * (self.k1 + 1)) / denom
                scores[i] += score

        ranked = sorted(
            enumerate(scores),
            key=lambda x: x[1],
            reverse=True,
        )
        results: list[tuple[Document, float]] = []
        for idx, s in ranked[: max(0, int(top_k))]:
            if s <= 0:
                continue
            results.append((index.documents[idx], float(s)))
        return results
