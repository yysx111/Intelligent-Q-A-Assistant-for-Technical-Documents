from langchain_community.embeddings import HuggingFaceEmbeddings
from app.config import get_settings

def get_embeddings():
    """获取BGE嵌入模型"""
    settings = get_settings()
    
    model_kwargs = {
        'device': 'cpu',  # 或 'cpu'
        'trust_remote_code': True
    }
    
    encode_kwargs = {
        'normalize_embeddings': True,
        'show_progress_bar': True
    }
    
    embeddings = HuggingFaceEmbeddings(
        model_name=settings.embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings