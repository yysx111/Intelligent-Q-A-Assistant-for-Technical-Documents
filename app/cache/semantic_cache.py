from typing import Optional, Tuple
import numpy as np
from app.rag.embeddings import get_embeddings
from app.config import get_settings

class SemanticCache:
    """基于Embedding相似度的语义缓存"""
    
    def __init__(self):
        self.settings = get_settings()
        self.embeddings = get_embeddings()
        self.cache = {}  # {query_embedding: (answer, metadata)}
        self.max_size = self.settings.cache_max_size
        self.threshold = self.settings.cache_similarity_threshold
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """获取文本嵌入"""
        embedding = self.embeddings.embed_query(text)
        return np.array(embedding)
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def get(self, query: str) -> Optional[dict]:
        """查询缓存"""
        if not self.settings.cache_enabled:
            return None
        
        query_embedding = self._get_embedding(query)
        
        for cached_query, (answer, metadata) in self.cache.items():
            cached_embedding = self._get_embedding(cached_query)
            similarity = self._cosine_similarity(query_embedding, cached_embedding)
            
            if similarity >= self.threshold:
                return {
                    "answer": answer,
                    "metadata": metadata,
                    "similarity": similarity,
                    "cache_hit": True
                }
        
        return None
    
    def set(self, query: str, answer: str, metadata: dict):
        """设置缓存"""
        if not self.settings.cache_enabled:
            return
        
        if len(self.cache) >= self.max_size:
            # 简单的LRU策略：删除最旧的
            self.cache.pop(next(iter(self.cache)))
        
        self.cache[query] = (answer, metadata)
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()