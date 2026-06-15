from qdrant_client import QdrantClient, models

from src.common.bm25 import bm25_vector
from src.common.config import settings
from src.common.embedding import embed_dense
from src.retrieval.bm25_retriever import colbert_rerank
from src.retrieval.cross_encoder_rerank import cross_encoder_rerank
from src.schema.agent_schemas import Evidence


def hybrid_search(
    query: str,
    dense_st,
    colbert_model,
    cross_encoder,
    flt=None,
    collection_name: str = settings.collection_name,
    qdrant_url: str = settings.qdrant_url,
    top_n: int = 100,
    top_colbert: int = 50,
    top_k: int = 30,
    rerank: bool = True,
    use_dense: bool = True,
    use_sparse: bool = True,
):
    if not use_dense and not use_sparse:
        return []

    dense = (
        embed_dense([query], dense_st, is_query=True)
        if use_dense
        else None
    )

    client = QdrantClient(
        url=qdrant_url,
        prefer_grpc=True,
        timeout=settings.qdrant_timeout,
    )
    try:
        prefetch = []
        if use_dense:
            prefetch.append(
                models.Prefetch(
                    query=dense[0],
                    limit=top_n,
                    using="dense",
                    filter=flt,
                )
            )
        if use_sparse:
            prefetch.append(
                models.Prefetch(
                    query=bm25_vector(query),
                    limit=top_n,
                    using="sparse",
                    filter=flt,
                )
            )

        if len(prefetch) == 1:
            request = prefetch[0]
            points = client.query_points(
                collection_name=collection_name,
                query=request.query,
                using=request.using,
                query_filter=request.filter,
                limit=top_n,
                with_payload=True,
                with_vectors=False,
                timeout=settings.qdrant_timeout,
            ).points
        else:
            points = client.query_points(
                collection_name=collection_name,
                prefetch=prefetch,
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=top_n,
                with_payload=True,
                with_vectors=False,
                timeout=settings.qdrant_timeout,
            ).points
    finally:
        client.close()

    candidates = [
        Evidence(
            unit_id=point.payload["unit_id"],
            chunk_id=point.payload.get("chunk_id"),
            text=point.payload["text"],
            source="hybrid",
            score=point.score,
            metadata=point.payload,
        )
        for point in points
    ]
    if not rerank:
        return candidates

    candidates = colbert_rerank(
        query, candidates, colbert_model, top_k=top_colbert
    )
    return cross_encoder_rerank(
        query, candidates, cross_encoder, top_k=top_k
    )

if __name__ == "__main__":
    from src.common.embedding import load_dense_model, load_colbert_model
    from dotenv import load_dotenv

    load_dotenv()

    dense_st = load_dense_model()
    colbert_model = load_colbert_model()

    query = "Doanh nghiệp nhỏ và vừa được hỗ trợ những gì theo quy định"
    candidates = hybrid_search(
        query,
        dense_st,
        colbert_model,
        None,
    )
    print(type(candidates[0]))
