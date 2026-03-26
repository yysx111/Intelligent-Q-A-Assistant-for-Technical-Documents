from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from app.rag.chain import RAGChain
from app.cache.semantic_cache import SemanticCache
from app.rag.document_loader import load_document_from_bytes
from app.rag.text_splitter import split_documents
from app.rag.vector_store import get_vector_store
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

_rag_chain: Optional[RAGChain] = None
_rag_lock = asyncio.Lock()


async def get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is not None:
        return _rag_chain
    async with _rag_lock:
        if _rag_chain is None:
            _rag_chain = RAGChain()
    return _rag_chain

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
    rag_chain = None
    original_cache = None
    try:
        rag_chain = await get_rag_chain()
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
    finally:
        if rag_chain is not None and original_cache is not None:
            rag_chain.cache = original_cache

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """上传文档并写入索引"""
    try:
        rag_chain = await get_rag_chain()
        raw = await file.read()
        docs = load_document_from_bytes(file.filename or "uploaded", raw, source=file.filename)
        chunks = split_documents(docs)
        vector_store = rag_chain.hybrid_search.vector_store if hasattr(rag_chain, "hybrid_search") else get_vector_store()
        vector_store.add_documents(chunks)
        return {"message": f"已索引 {len(chunks)} 个片段"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/cache/clear")
async def clear_cache():
    """清空缓存"""
    rag_chain = await get_rag_chain()
    rag_chain.cache.clear()
    return {"message": "缓存已清空"}

@app.get("/health")
async def health_check():
    """健康检查"""
    rag_chain = await get_rag_chain()
    cache_size = rag_chain.cache.collection.count()
    return {"status": "healthy", "cache_size": int(cache_size)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
