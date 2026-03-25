from langchain_chroma import Chroma
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import chromadb
from app.config import get_settings
from app.rag.embeddings import get_embeddings

def get_vector_store():
    """获取或创建向量存储"""
    settings = get_settings()
    embeddings = get_embeddings()
    
    # 创建持久化客户端
    client = chromadb.PersistentClient(
        path=settings.chroma_persist_dir
    )
    
    # 创建或获取collection
    collection = client.get_or_create_collection(
        name="technical_docs",
        metadata={"hnsw:space": "cosine"}
    )
    
    # 创建Chroma向量存储
    vector_store = Chroma(
        client=client,
        collection_name="technical_docs",
        embedding_function=embeddings
    )
    
    return vector_store

def add_documents(documents: list[Document]):
    """添加文档到向量库"""
    vector_store = get_vector_store()
    vector_store.add_documents(documents)