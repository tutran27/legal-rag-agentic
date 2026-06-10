
class Settings:
    qdrant_url = "http://localhost:6333"
    collection_name = "legal_agent_rag"
    top_n = 50
    top_k = 10
    batch_size = 32
    normalize_dense = True
    
    dense_model = "tutran27/vietnamese-legal-phapdien-embedding-v1"
    bge_model = "BAAI/bge-m3"
    
settings=Settings()
    
