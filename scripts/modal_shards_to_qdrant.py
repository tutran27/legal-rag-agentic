from __future__ import annotations

import json
import uuid
from pathlib import Path

import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.common.bm25 import average_length, bm25_vector
from src.common.config import settings
from src.indexing.build_hybrid_ingest import ensure_collection

SHARDS = Path("data/embedding_shards")
CHECKPOINT = SHARDS / "qdrant_checkpoint.json"


def main(recreate: bool = False) -> None:
    files = sorted(SHARDS.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy shard trong {SHARDS}")

    client = QdrantClient(url=settings.qdrant_url, prefer_grpc=True)
    checkpoint = {"shard": 0}
    if CHECKPOINT.exists() and not recreate:
        checkpoint.update(json.loads(CHECKPOINT.read_text()))

    first = pq.read_table(files[0], columns=["dense"])
    dense_dim = len(first.column("dense")[0].as_py())
    ensure_collection(
        client,
        settings.collection_name,
        dense_dim=dense_dim,
        recreate=recreate,
    )

    avg_length = average_length(
        row["text"]
        for file in files
        for batch in pq.ParquetFile(file).iter_batches(
            batch_size=4096, columns=["text"]
        )
        for row in batch.to_pylist()
    )

    for shard_index, file in enumerate(
        tqdm(files[checkpoint["shard"] :], desc="Ingest shards"),
        start=checkpoint["shard"],
    ):
        rows = pq.read_table(file).to_pylist()
        points = [
            models.PointStruct(
                id=str(
                    uuid.uuid5(
                        uuid.NAMESPACE_URL,
                        f"legal-rag:{row['chunk_id']}",
                    )
                ),
                vector={
                    "dense": row.pop("dense"),
                    "sparse": bm25_vector(row["text"], avg_length),
                },
                payload=row,
            )
            for row in rows
        ]
        client.upload_points(
            collection_name=settings.collection_name,
            points=points,
            batch_size=128,
            parallel=4,
            wait=False,
        )
        checkpoint["shard"] = shard_index + 1
        CHECKPOINT.write_text(json.dumps(checkpoint))

    client.update_collection(
        collection_name=settings.collection_name,
        hnsw_config=models.HnswConfigDiff(m=16, on_disk=True),
    )
    client.close()


if __name__ == "__main__":
    main()
