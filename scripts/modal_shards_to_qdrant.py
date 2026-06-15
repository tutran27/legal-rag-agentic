from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import pyarrow.parquet as pq
from qdrant_client import QdrantClient, models
from tqdm import tqdm

from src.common.bm25 import average_length, bm25_vector
from src.common.config import settings
from src.indexing.qdrant_collection import create_collection

DEFAULT_SHARDS = Path("data/embedding_shards")


def main(
    recreate: bool = False,
    shards_dir: str | Path = DEFAULT_SHARDS,
    build_hnsw: bool = False,
) -> None:
    shards = Path(shards_dir)
    checkpoint_path = shards / "qdrant_checkpoint.json"
    files = sorted(shards.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"Không tìm thấy shard trong {shards}")

    client = QdrantClient(url=settings.qdrant_url, prefer_grpc=True)
    collection_existed = client.collection_exists(settings.collection_name)
    checkpoint = {
        "shard": 0,
        "collection": settings.collection_name,
    }
    if recreate:
        checkpoint_path.unlink(missing_ok=True)
    if checkpoint_path.exists() and not recreate and collection_existed:
        saved_checkpoint = json.loads(checkpoint_path.read_text())
        if saved_checkpoint.get("collection") == settings.collection_name:
            checkpoint.update(saved_checkpoint)

    first = pq.read_table(files[0], columns=["dense"])
    dense_dim = len(first.column("dense")[0].as_py())
    create_collection(
        client,
        settings.collection_name,
        dense_dim=dense_dim,
        recreate=recreate,
    )
    client.update_collection(
        collection_name=settings.collection_name,
        hnsw_config=models.HnswConfigDiff(m=0),
        optimizer_config=models.OptimizersConfigDiff(
            indexing_threshold=0,
            max_optimization_threads=1,
        ),
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
        parquet = pq.ParquetFile(file)
        for batch in parquet.iter_batches(batch_size=256):
            rows = batch.to_pylist()
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
                batch_size=64,
                parallel=1,
                wait=True,
            )
        checkpoint["shard"] = shard_index + 1
        checkpoint_path.write_text(json.dumps(checkpoint))

    if build_hnsw:
        client.update_collection(
            collection_name=settings.collection_name,
            hnsw_config=models.HnswConfigDiff(m=16, on_disk=True),
            optimizer_config=models.OptimizersConfigDiff(
                indexing_threshold=20_000,
                max_optimization_threads=1,
            ),
        )
    client.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--recreate", action="store_true")
    parser.add_argument(
        "--shards-dir",
        default=str(DEFAULT_SHARDS),
    )
    parser.add_argument("--build-hnsw", action="store_true")
    args = parser.parse_args()
    main(
        recreate=args.recreate,
        shards_dir=args.shards_dir,
        build_hnsw=args.build_hnsw,
    )
