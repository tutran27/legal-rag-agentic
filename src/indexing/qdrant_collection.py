from qdrant_client import QdrantClient, models


def validate_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int,
) -> None:
    params = client.get_collection(collection_name).config.params
    dense = params.vectors.get("dense")
    sparse = (params.sparse_vectors or {}).get("sparse")

    if dense is None or dense.size != dense_dim:
        raise RuntimeError(
            f"Collection {collection_name} không có dense vector "
            f"{dense_dim} chiều. Hãy chạy lại với --recreate."
        )
    if sparse is None or sparse.modifier != models.Modifier.IDF:
        raise RuntimeError(
            f"Collection {collection_name} chưa bật sparse IDF. "
            "Hãy chạy lại với --recreate."
        )


def create_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int,
    recreate: bool = False,
) -> None:
    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    if client.collection_exists(collection_name):
        validate_collection(client, collection_name, dense_dim)
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
                on_disk=True,
            )
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(
                index=models.SparseIndexParams(on_disk=True),
                modifier=models.Modifier.IDF,
            )
        },
        hnsw_config=models.HnswConfigDiff(
            m=0,
            on_disk=True,
        ),
        optimizers_config=models.OptimizersConfigDiff(
            indexing_threshold=0,
            max_optimization_threads=1,
        ),
        on_disk_payload=True,
    )
