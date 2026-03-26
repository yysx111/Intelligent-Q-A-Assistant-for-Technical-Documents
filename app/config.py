from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    dashscope_api_key: str
    qwen_model: str = "qwen-turbo"
    
    chroma_persist_dir: str = "./data/chroma"
    
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.88
    cache_max_size: int = 1000
    cache_collection_name: str = "semantic_cache"
    
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    rerank_model: str = "BAAI/bge-reranker-large"
    rerank_top_k: int = 5
    
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    return Settings()
