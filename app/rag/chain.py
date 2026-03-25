from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from app.rag.hybrid_search import HybridSearch
from app.llm.qwen_client import QwenClient
from app.cache.semantic_cache import SemanticCache
from app.models.schemas import QueryResponse
import json

class RAGChain:
    """RAG问答链"""
    
    def __init__(self):
        self.hybrid_search = HybridSearch()
        self.llm_client = QwenClient()
        self.cache = SemanticCache()
    
    async def query(self, question: str) -> QueryResponse:
        """执行问答"""
        # 1. 检查缓存
        cached_result = self.cache.get(question)
        if cached_result:
            return QueryResponse(
                answer=cached_result["answer"],
                code_examples=[],
                references=[],
                confidence=cached_result["similarity"],
                cache_hit=True
            )
        
        # 2. 混合检索
        search_results = self.hybrid_search.search(question, top_k=5)
        
        # 3. 构建上下文
        context = "\n\n".join([doc for doc, score, meta in search_results])
        
        # 4. 构建Prompt（要求结构化输出）
        prompt = self._build_prompt(question, context, search_results)
        
        # 5. 调用LLM获取结构化答案
        output_schema = {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "code_examples": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "language": {"type": "string"},
                            "code": {"type": "string"},
                            "description": {"type": "string"}
                        }
                    }
                },
                "references": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "source": {"type": "string"},
                            "relevance_score": {"type": "number"}
                        }
                    }
                },
                "confidence": {"type": "number"}
            },
            "required": ["answer", "confidence"]
        }
        
        try:
            llm_response = self.llm_client.generate_structured(
                prompt, 
                output_schema=output_schema
            )
            
            # 6. 构建响应
            response = QueryResponse(
                answer=llm_response.get("answer", ""),
                code_examples=[
                    {"language": ex.get("language", "python"), 
                     "code": ex.get("code", ""), 
                     "description": ex.get("description", "")}
                    for ex in llm_response.get("code_examples", [])
                ],
                references=[
                    {"source": ref.get("source", ""), 
                     "relevance_score": ref.get("relevance_score", 0),
                     "content": search_results[i][0] if i < len(search_results) else ""}
                    for i, ref in enumerate(llm_response.get("references", []))
                ],
                confidence=llm_response.get("confidence", 0.5),
                cache_hit=False
            )
            
            # 7. 缓存结果
            self.cache.set(question, response.answer, {
                "code_examples": response.code_examples,
                "references": response.references
            })
            
            return response
            
        except Exception as e:
            # 降级处理
            return QueryResponse(
                answer=f"抱歉，处理您的问题时出现错误：{str(e)}",
                code_examples=[],
                references=[],
                confidence=0.0,
                cache_hit=False
            )
    
    def _build_prompt(self, question: str, context: str, search_results: list) -> str:
        """构建Prompt"""
        references = "\n".join([
            f"- {meta.get('source', '未知')} (相关度: {score:.2f})"
            for doc, score, meta in search_results
        ])
        
        return f"""基于以下技术文档上下文回答问题。

## 上下文信息
{context}

## 参考来源
{references}

## 问题
{question}

## 要求
1. 提供准确、详细的答案
2. 如果涉及代码，提供完整的代码示例
3. 标注参考来源
4. 评估答案的置信度(0-1)

请严格按照JSON格式返回答案。"""