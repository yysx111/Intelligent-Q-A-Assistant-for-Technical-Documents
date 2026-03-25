from FlagEmbedding import FlagReranker
from app.config import get_settings

class BGEReranker:
    """BGE重排序器"""
    
    def __init__(self):
        settings = get_settings()
        self.reranker = FlagReranker(
            settings.rerank_model,
            use_fp16=False  # 显存不足可改为True
        )
        self.top_k = settings.rerank_top_k
    
    def rerank(self, query: str, documents: list[str]) -> list[tuple[str, float]]:
        """对文档进行重排序"""
        if not documents:
            return []
        
        # 计算分数
        scores = self.reranker.compute_score(
            [[query, doc] for doc in documents],
            normalize=True
        )
        
        # 配对文档和分数
        doc_score_pairs = list(zip(documents, scores))
        
        # 按分数降序排序
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)
        
        # 返回top_k
        return doc_score_pairs[:self.top_k]