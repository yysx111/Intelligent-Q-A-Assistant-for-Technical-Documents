from pydantic import BaseModel, Field
from typing import List, Optional

class SourceDocument(BaseModel):
    """参考文档来源"""
    content: str = Field(..., description="文档内容片段")
    source: str = Field(..., description="文档来源URL或路径")
    relevance_score: float = Field(..., description="相关度分数")

class CodeExample(BaseModel):
    """代码示例"""
    language: str = Field(..., description="编程语言")
    code: str = Field(..., description="代码内容")
    description: str = Field(..., description="代码说明")

class QueryResponse(BaseModel):
    """结构化问答响应"""
    answer: str = Field(..., description="问题的答案")
    code_examples: List[CodeExample] = Field(default_factory=list, description="相关代码示例")
    references: List[SourceDocument] = Field(default_factory=list, description="参考文档")
    confidence: float = Field(..., description="答案置信度(0-1)")
    cache_hit: bool = Field(default=False, description="是否命中缓存")