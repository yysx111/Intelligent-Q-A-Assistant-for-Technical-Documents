import time
import uuid
from typing import Optional
import numpy as np
import chromadb
from app.rag.embeddings import get_embeddings
from app.config import get_settings

class SemanticCache:
    def __init__(self):
        self.settings = get_settings()
        self.embeddings = get_embeddings()
        self.threshold = self.settings.cache_similarity_threshold
        self.max_size = self.settings.cache_max_size
        self.client = chromadb.PersistentClient(path=self.settings.chroma_persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=self.settings.cache_collection_name,
            metadata={"hnsw:space": "cosine"}
        )
    
    def _get_embedding(self, text: str) -> list[float]:
        return list(self.embeddings.embed_query(text))
    
    def get(self, query: str) -> Optional[dict]:
        if not self.settings.cache_enabled:
            return None
        embedding = self._get_embedding(query)
        result = self.collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "metadatas", "distances", "ids"]
        )
        ids = result.get("ids", [[]])[0]
        if not ids:
            return None
        distance = result.get("distances", [[1.0]])[0][0]
        similarity = 1.0 - float(distance)
        if similarity < self.threshold:
            return None
        doc = result.get("documents", [[""]])[0][0]
        meta = result.get("metadatas", [[{}]])[0][0] or {}
        meta["last_access"] = time.time()
        meta["hit_count"] = int(meta.get("hit_count", 0)) + 1
        self.collection.update(ids=[ids[0]], metadatas=[meta])
        return {
            "answer": doc,
            "metadata": meta,
            "similarity": similarity,
            "cache_hit": True
        }
    
    def _evict_if_needed(self):
        count = self.collection.count()
        if count < self.max_size:
            return
        data = self.collection.get(include=["metadatas", "ids"])
        ids = data.get("ids", [])
        metas = data.get("metadatas", [])
        if not ids:
            return
        oldest_idx = 0
        oldest_ts = float("inf")
        for i, m in enumerate(metas):
            ts = float(m.get("last_access", m.get("created_at", time.time())))
            if ts < oldest_ts:
                oldest_ts = ts
                oldest_idx = i
        self.collection.delete(ids=[ids[oldest_idx]])
    
    def set(self, query: str, answer: str, metadata: dict):
        if not self.settings.cache_enabled:
            return
        self._evict_if_needed()
        doc_id = uuid.uuid4().hex
        now = time.time()
        meta = {
            "question": query,
            "created_at": now,
            "last_access": now,
            "hit_count": 0
        }
        if isinstance(metadata, dict):
            meta.update(metadata)
        embedding = self._get_embedding(query)
        self.collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[answer],
            metadatas=[meta]
        )
    
    def clear(self):
        data = self.collection.get(include=["ids"])
        ids = data.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
