from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    # 通义千问配置
    dashscope_api_key: str
    qwen_model: str = "qwen-turbo"
    
    # 向量数据库
    chroma_persist_dir: str = "./data/chroma"
    
    # 模型配置
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    rerank_model: str = "BAAI/bge-reranker-large"
    rerank_top_k: int = 5
    
    # 应用配置
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()