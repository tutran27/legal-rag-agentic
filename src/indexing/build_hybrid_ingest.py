from __future__ import annotations

import uuid

import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.common.config import settings
from src.common.embedding import embed_text, load_model


def to_sparse_vector(values: dict) -> models.SparseVector:
    items = sorted(
        (int(index), float(weight))
        for index, weight in values.items()
        if weight
    )
    return models.SparseVector(
        indices=[index for index, _ in items],
        values=[weight for _, weight in items],
    )


def create_collection(
    client: QdrantClient,
    collection_name: str = settings.collection_name,
    dense_dim: int=1024,
    colbert_dim: int = 1024,
) -> None:
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)
        
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=dense_dim,
                distance=models.Distance.COSINE,
            ),
            "colbert": models.VectorParams(
                size=colbert_dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )


def ingest_hybrid_qdrant_index(
    client: QdrantClient,
    corpus_path: str = "data/processed/retrieval_corpus.parquet",
    collection_name: str = settings.collection_name,
    batch_size: int = settings.batch_size,
    recreate_collection: bool = False,
) -> None:
    dense_model, bge_model = load_model()
    parquet = pq.ParquetFile(corpus_path)

    if recreate_collection and client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    collection_exists = client.collection_exists(collection_name)
    progress = tqdm(
        total=parquet.metadata.num_rows,
        desc="Ingest Qdrant",
        unit="chunk",
    )

    for batch in parquet.iter_batches(batch_size=batch_size):
        rows = batch.to_pylist()
        texts = [row["text"] for row in rows]
        dense, sparse, colbert = embed_text(
            texts,
            dense_model,
            bge_model,
            batch_size=batch_size,
        )

        if not collection_exists:
            create_collection(
                client,
                collection_name,
                dense_dim=len(dense[0]),
                colbert_dim=len(colbert[0][0]),
            )
            collection_exists = True

        points = []
        for index, row in enumerate(rows):
            points.append(
                models.PointStruct(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"legal-rag:{row['chunk_id']}",
                        )
                    ),
                    vector={
                        "dense": dense[index].tolist(),
                        "sparse": to_sparse_vector(sparse[index]),
                        "colbert": colbert[index].tolist(),
                    },
                    payload=row,
                )
            )

        client.upsert(
            collection_name=collection_name,
            points=points,
            wait=True,
        )
        progress.update(len(rows))

    progress.close()
    print(f"Ingested {parquet.metadata.num_rows} chunks into {collection_name}")


if __name__ == "__main__":
    try:
        print(f"Connecting to Qdrant at {settings.qdrant_url}")
        client=QdrantClient(url=settings.qdrant_url)
        
        print(f"Creating collection {settings.collection_name}")
        create_collection(
            client,
            collection_name=settings.collection_name,
        )
        
        print(f"Ingesting data into collection {settings.collection_name}")
        ingest_hybrid_qdrant_index(
            client,
            recreate_collection=True,
        )
    finally:
        client.close()
