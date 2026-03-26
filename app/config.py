from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", case_sensitive=False)

    dashscope_api_key: str = ""
    qwen_model: str = "qwen-turbo"
    max_retries: int = 3
    retry_delay: float = 1.0
    
    chroma_persist_dir: str = "./data/chroma"
    
    cache_enabled: bool = True
    cache_similarity_threshold: float = 0.88
    cache_max_size: int = 1000
    cache_collection_name: str = "semantic_cache"
    
    embedding_model: str = "BAAI/bge-large-zh-v1.5"
    rerank_model: str = "BAAI/bge-reranker-large"
    rerank_top_k: int = 5

    hybrid_alpha: float = 0.7
    vector_search_k: int = 10
    keyword_search_k: int = 10
    
    app_host: str = "0.0.0.0"
    app_port: int = 8000

@lru_cache()
def get_settings() -> Settings:
    return Settings()
