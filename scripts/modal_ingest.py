from __future__ import annotations

import json
from pathlib import Path

import modal

APP_NAME = "legal-rag-embedding"
VOLUME_NAME = "legal-rag-ingest-data"
CORPUS = "/data/retrieval_corpus.parquet"
OUTPUT_DIR = Path("/data/embedding_shards")
CHECKPOINT = Path("/data/embedding_checkpoint.json")
BATCH_SIZES = (8192, 6144, 4096, 3072, 2048, 1536, 1024, 768, 512, 256, 128)

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
secret = modal.Secret.from_name("legal-rag-secrets")

image = modal.Image.debian_slim(python_version="3.11").uv_pip_install(
    "torch",
    "sentence-transformers",
    "huggingface-hub",
    "pyarrow",
)


@app.function(
    image=image,
    gpu="A100-80GB",
    cpu=16,
    memory=65536,
    timeout=24 * 60 * 60,
    retries=modal.Retries(max_retries=5, initial_delay=10),
    volumes={"/data": volume},
    secrets=[secret],
)
def export_embeddings():
    import os
    import time

    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    from sentence_transformers import SentenceTransformer

    parquet = pq.ParquetFile(CORPUS)
    checkpoint = {"row_group": 0, "batch_size": BATCH_SIZES[0]}
    if CHECKPOINT.exists():
        checkpoint.update(json.loads(CHECKPOINT.read_text()))
    checkpoint["batch_size"] = BATCH_SIZES[0]

    model = SentenceTransformer(
        os.getenv(
            "DENSE_MODEL",
            "tutran27/vietnamese-legal-phapdien-embedding-v1",
        ),
        token=os.getenv("HF_TOKEN"),
        device="cuda",
        model_kwargs={"torch_dtype": torch.float16},
    )
    OUTPUT_DIR.mkdir(exist_ok=True)
    started = time.monotonic()

    for row_group in range(checkpoint["row_group"], parquet.num_row_groups):
        table = parquet.read_row_group(row_group)
        texts = table.column("text").to_pylist()

        start_index = BATCH_SIZES.index(checkpoint["batch_size"])
        for batch_size in BATCH_SIZES[start_index:]:
            try:
                dense = model.encode(
                    texts,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
                checkpoint["batch_size"] = batch_size
                break
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
        else:
            raise RuntimeError("Không đủ VRAM với batch 128")

        dense_array = pa.array(
            dense.astype("float16").tolist(),
            type=pa.list_(pa.float16(), dense.shape[1]),
        )
        shard = table.append_column("dense", dense_array)
        pq.write_table(
            shard,
            OUTPUT_DIR / f"part-{row_group:04d}.parquet",
            compression="zstd",
        )

        checkpoint["row_group"] = row_group + 1
        CHECKPOINT.write_text(json.dumps(checkpoint))
        volume.commit()
        print(
            f"row_group={row_group + 1}/{parquet.num_row_groups} "
            f"batch={checkpoint['batch_size']} "
            f"elapsed={time.monotonic() - started:.0f}s"
        )

    checkpoint["completed"] = True
    CHECKPOINT.write_text(json.dumps(checkpoint))
    volume.commit()
    return checkpoint


@app.function(image=image, timeout=600, volumes={"/data": volume})
def reset():
    import shutil

    shutil.rmtree(OUTPUT_DIR, ignore_errors=True)
    CHECKPOINT.unlink(missing_ok=True)
    volume.commit()


@app.local_entrypoint()
def main(
    action: str = "start",
    corpus: str = "data/processed/retrieval_corpus.parquet",
    recreate: bool = False,
):
    if action == "upload":
        with volume.batch_upload(force=True) as upload:
            upload.put_file(corpus, "retrieval_corpus.parquet")
        print(f"Đã upload {corpus} lên {VOLUME_NAME}")
        return

    if action in {"start", "run"}:
        if recreate:
            reset.remote()
        if action == "start":
            call = export_embeddings.spawn()
            print(f"Đã chạy job: {call.object_id}")
        else:
            print(export_embeddings.remote())
        return

    raise ValueError("action phải là upload, start hoặc run")
