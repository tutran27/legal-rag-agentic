from http import client
from common.config import Settings
from qdrant_client import QdrantClient, models
from indexing.build_hybrid_ingest import embed_text

from src.common import settings
def hybrid_search(
    query: str,
    dense_st,
    bge_m3,
    flt = None, 
    collection_name: str = "legal_articles_hybrid",
    qdrant_url: str = "http://localhost:6333",
    top_n: int = 100,
    top_k: int = 80,
):
        client = QdrantClient(url=qdrant_url)
        dense_embedding, sparse_embedding, colbert_embedding = embed_text(
            [query],
            dense_st,
            bge_m3,
        )
        
        result=client.query_points(
        collection_name=collection_name,
        prefetch=[models.Prefetch(
            query_vector=dense_embedding[0],
            limit=top_n,
            using="dense",
            filter=flt,
        ),
     
        models.Prefetch(
            query_vector=sparse_embedding[0].tolist(),
            limit=top_n,
            using="sparse",
            filter=flt,
        )],
        query=models.FusionQuery(
            fusion=models.RRF
        ),
        limit=top_k,
        with_payload=True,
        with_vector=False
    )
        return result.points
   
if __name__ == "__main__":
    try:
        query = "Doanh nghiệp SME có thể có bao nhiêu người làm việc?"
        results = hybrid_search(
            query=query,
            dense_st=settings.dense_model,
            bge_m3=settings.bge_model,
            qdrant_url=settings.qdrant_url,
            collection_name=settings.collection_name,
            top_n=50,
            top_k=10,
        )
        for result in results[:5]:
            for k, v in result.payload.items():
                print(f"{k.upper()}: {v}")
            
    finally:
        client.close()
