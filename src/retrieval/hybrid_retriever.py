from qdrant_client import QdrantClient, models

from src.common.bm25 import bm25_vector
from src.common.config import settings
from src.common.embedding import embed_dense
from src.retrieval.colbert_reranker import colbert_rerank
from src.retrieval.cross_encoder_rerank import cross_encoder_rerank
from src.retrieval.qdrant_payload import PAYLOAD_FIELDS, payload_to_evidence
from src.schema.agent_schemas import Evidence


def hybrid_search_batch(
    queries: list[str],
    dense_st,
    flt=None,
    collection_name: str = settings.collection_name,
    qdrant_url: str = settings.qdrant_url,
    top_n: int = 100,
    use_dense: bool = True,
    use_sparse: bool = True,
    client: QdrantClient | None = None,
    dense_vectors=None,
    filters: list | None = None,
) -> list[list[Evidence]]:
    if not queries or (not use_dense and not use_sparse):
        return [[] for _ in queries]

    if use_dense and dense_vectors is None:
        dense_vectors = embed_dense(queries, dense_st, is_query=True)
    if not use_dense:
        dense_vectors = [None] * len(queries)
    requests = []
    search_params = models.SearchParams(
        hnsw_ef=settings.qdrant_hnsw_ef,
        exact=False,
        indexed_only=True,
    )
    query_filters = filters or [flt] * len(queries)
    if len(query_filters) != len(queries):
        raise ValueError("Số filter phải bằng số query.")

    for query, dense_vector, query_filter in zip(
        queries,
        dense_vectors,
        query_filters,
    ):
        prefetch = []
        if use_dense:
            prefetch.append(
                models.Prefetch(
                    query=dense_vector.tolist(),
                    limit=top_n,
                    using="dense",
                    filter=query_filter,
                    params=search_params,
                )
            )
        if use_sparse:
            prefetch.append(
                models.Prefetch(
                    query=bm25_vector(query),
                    limit=top_n,
                    using="sparse",
                    filter=query_filter,
                )
            )

        if len(prefetch) == 1:
            request = prefetch[0]
            requests.append(
                models.QueryRequest(
                    query=request.query,
                    using=request.using,
                    filter=request.filter,
                    params=request.params,
                    limit=top_n,
                    with_payload=PAYLOAD_FIELDS,
                    with_vector=False,
                )
            )
        else:
            requests.append(
                models.QueryRequest(
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=top_n,
                    with_payload=PAYLOAD_FIELDS,
                    with_vector=False,
                )
            )

    owns_client = client is None
    client = client or QdrantClient(
        url=qdrant_url,
        prefer_grpc=True,
        timeout=settings.qdrant_timeout,
    )
    try:
        responses = client.query_batch_points(
            collection_name=collection_name,
            requests=requests,
            timeout=settings.qdrant_timeout,
        )
    finally:
        if owns_client:
            client.close()

    return [
        [
            payload_to_evidence(
                point.payload,
                source="hybrid",
                score=point.score,
            )
            for point in response.points
        ]
        for response in responses
    ]


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
    candidates = hybrid_search_batch(
        [query],
        dense_st=dense_st,
        flt=flt,
        collection_name=collection_name,
        qdrant_url=qdrant_url,
        top_n=top_n,
        use_dense=use_dense,
        use_sparse=use_sparse,
    )[0]
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
