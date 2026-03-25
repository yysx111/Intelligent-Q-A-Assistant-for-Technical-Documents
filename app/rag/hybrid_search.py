from typing import List, Tuple
from app.rag.vector_store import get_vector_store
from app.rag.keyword_search import BM25Search
from app.rag.reranker import BGEReranker
from app.config import get_settings
import numpy as np

class HybridSearch:
    """混合检索（向量+关键词+Rerank）"""
    
    def __init__(self):
        self.settings = get_settings()
        self.vector_store = get_vector_store()
        self.bm25 = BM25Search()
        self.reranker = BGEReranker()
        self.alpha = self.settings.hybrid_alpha  # 向量检索权重
    
    def search(self, query: str, top_k: int = 10) -> List[Tuple[str, float, dict]]:
        """
        混合检索
        返回: [(文档内容, 综合分数, 元数据), ...]
        """
        # 1. 向量检索
        vector_results = self.vector_store.similarity_search_with_score(
            query, 
            k=self.settings.vector_search_k
        )
        
        # 2. 关键词检索
        keyword_results = self.bm25.search(query, top_k=self.settings.keyword_search_k)
        
        # 3. 归一化分数并融合
        all_docs = {}
        
        # 处理向量检索结果
        for doc, score in vector_results:
            doc_id = id(doc)
            all_docs[doc_id] = {
                'content': doc.page_content,
                'metadata': doc.metadata,
                'vector_score': 1 - score,  # 转换为相似度
                'keyword_score': 0.0
            }
        
        # 处理关键词检索结果
        for doc, score in keyword_results:
            doc_id = id(doc)
            if doc_id in all_docs:
                all_docs[doc_id]['keyword_score'] = score
            else:
                all_docs[doc_id] = {
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'vector_score': 0.0,
                    'keyword_score': score
                }
        
        # 4. 计算综合分数
        for doc_id, data in all_docs.items():
            data['combined_score'] = (
                self.alpha * data['vector_score'] + 
                (1 - self.alpha) * data['keyword_score']
            )
        
        # 5. 排序
        sorted_docs = sorted(
            all_docs.values(),
            key=lambda x: x['combined_score'],
            reverse=True
        )[:top_k]
        
        # 6. Rerank重排序
        doc_contents = [d['content'] for d in sorted_docs]
        reranked = self.reranker.rerank(query, doc_contents)
        
        # 7. 构建最终结果
        results = []
        for content, score in reranked:
            for doc_data in sorted_docs:
                if doc_data['content'] == content:
                    results.append((content, score, doc_data['metadata']))
                    break
        
        return results