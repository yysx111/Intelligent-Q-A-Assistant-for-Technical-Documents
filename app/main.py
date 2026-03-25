from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from app.rag.chain import RAGChain
from app.cache.semantic_cache import SemanticCache
import asyncio

app = FastAPI(
    title="智能技术文档问答助手",
    description="基于RAG的技术文档问答系统",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化RAG链
rag_chain = RAGChain()
cache = SemanticCache()

class QueryRequest(BaseModel):
    query: str
    use_rerank: bool = True
    use_cache: bool = True

class QueryResponse(BaseModel):
    answer: str
    code_examples: list
    references: list
    confidence: float
    cache_hit: bool

@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """问答接口"""
    try:
        # 如果禁用缓存，临时清空
        if not request.use_cache:
            original_cache = rag_chain.cache
            rag_chain.cache = SemanticCache()
            rag_chain.cache.settings.cache_enabled = False
        
        result = await rag_chain.query(request.query)
        
        return QueryResponse(
            answer=result.answer,
            code_examples=result.code_examples,
            references=[{
                "source": ref.source,
                "relevance_score": ref.relevance_score,
                "content": ref.content
            } for ref in result.references],
            confidence=result.confidence,
            cache_hit=result.cache_hit
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cache/clear")
async def clear_cache():
    """清空缓存"""
    cache.clear()
    return {"message": "缓存已清空"}

@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "cache_size": len(cache.cache)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)