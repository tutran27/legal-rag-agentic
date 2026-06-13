from __future__ import annotations

import json
import os
import uuid
from typing import Any

import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.common.config import settings
from src.common.embedding import embed_text, load_model


DEFAULT_UPSERT_BATCH_SIZE = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "1"))
DEFAULT_MAX_COLBERT_VECTORS = int(os.getenv("COLBERT_MAX_VECTORS", "64"))


def to_sparse_vector(values: Any) -> models.SparseVector:
    if isinstance(values, models.SparseVector):
        return values

    if isinstance(values, dict) and "indices" in values and "values" in values:
        return models.SparseVector(
            indices=[int(i) for i in values["indices"]],
            values=[float(v) for v in values["values"]],
        )

    if isinstance(values, dict):
        items = [(int(i), float(v)) for i, v in values.items() if v]
        items.sort(key=lambda x: x[0])
        return models.SparseVector(
            indices=[i for i, _ in items],
            values=[v for _, v in items],
        )

    raise TypeError(f"Unsupported sparse vector format: {type(values)}")


def ensure_collection(
    client: QdrantClient,
    collection_name: str,
    dense_dim: int = settings.dense_dim,
    colbert_dim: int = settings.colbert_dim,
    recreate: bool = False,
) -> None:
    if recreate and client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    if client.collection_exists(collection_name):
        return

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(size=dense_dim, distance=models.Distance.COSINE),
            "colbert": models.VectorParams(
                size=colbert_dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM
                ),
            ),
        },
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )


def ingest_hybrid_qdrant_index(
    client: QdrantClient,
    corpus_path: str = "data/processed/retrieval_corpus.parquet",
    collection_name: str = settings.collection_name,
    batch_size: int = settings.batch_size,
    recreate_collection: bool = False,
    dense_model: Any | None = None,
    bge_model: Any | None = None,
) -> None:
    if dense_model is None or bge_model is None:
        dense_model, bge_model = load_model()
    parquet = pq.ParquetFile(corpus_path)

    collection_ready = client.collection_exists(collection_name) and not recreate_collection
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

        if not collection_ready:
            ensure_collection(
                client=client,
                collection_name=collection_name,
                dense_dim=len(dense[0]),
                colbert_dim=len(colbert[0][0]),
                recreate=recreate_collection,
            )
            collection_ready = True

        max_colbert_vectors = DEFAULT_MAX_COLBERT_VECTORS
        points: list[models.PointStruct] = []
        for i, row in enumerate(rows):
            payload = json.loads(json.dumps(row, ensure_ascii=False, default=str))
            colbert_vecs = colbert[i][:max_colbert_vectors] if max_colbert_vectors > 0 else colbert[i]
            colbert_payload = colbert_vecs.tolist() if hasattr(colbert_vecs, "tolist") else [list(v) for v in colbert_vecs]

            points.append(
                models.PointStruct(
                    id=str(
                        uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"legal-rag:{row['chunk_id']}",
                        )
                    ),
                    vector={
                        "dense": dense[i].tolist(),
                        "sparse": to_sparse_vector(sparse[i]),
                        "colbert": colbert_payload,
                    },
                    payload=payload,
                )
            )

        for start in range(0, len(points), DEFAULT_UPSERT_BATCH_SIZE):
            client.upsert(
                collection_name=collection_name,
                points=points[start:start + DEFAULT_UPSERT_BATCH_SIZE],
                wait=True,
            )
        progress.update(len(rows))

    progress.close()
    print(f"Ingested {parquet.metadata.num_rows} chunks into {collection_name}")


if __name__ == "__main__":
    try:
        print(f"Connecting to Qdrant at {settings.qdrant_url}")
        client=QdrantClient(url=settings.qdrant_url)
        ensure_collection(
            client=client,
            collection_name=settings.collection_name,
            recreate=True,
        )
        
        print("Loading embedding models")
        dense_model, bge_model = load_model()
        
        print(f"Ingesting data into collection {settings.collection_name}")
        ingest_hybrid_qdrant_index(
            client,
            recreate_collection=True,
            dense_model=dense_model,
            bge_model=bge_model,
        )
    finally:
        client.close()
