from __future__ import annotations

import json
from pathlib import Path

import modal


APP_NAME = "legal-rag-pipeline-test"
VOLUME_NAME = "legal-rag-ingest-data"
REMOTE_REPO = "/root/legal-agent-rag"
DEFAULT_QUERY = (
    "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ "
    "theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?"
)


app = modal.App(APP_NAME)
secret = modal.Secret.from_name("legal-rag-secrets")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .uv_pip_install(
        "accelerate",
        "bitsandbytes",
        "duckdb",
        "FlagEmbedding",
        "huggingface_hub",
        "jsonschema",
        "networkx",
        "orjson",
        "python-dotenv",
        "pandas",
        "polars",
        "pyarrow",
        "pydantic",
        "qdrant-client",
        "rank-bm25",
        "regex",
        "requests",
        "sentence-transformers",
        "torch",
        "tqdm",
        "transformers",
    )
    .add_local_dir(
        ".",
        remote_path=REMOTE_REPO,
        ignore=[
            ".git",
            ".env",
            ".venv",
            "__pycache__",
            ".pytest_cache",
            "data",
            "inference_errors.json",
            "results.json",
            "R2AIStage1DATA.json",
        ],
    )
)


@app.function(
    image=image,
    gpu="L4",
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    secrets=[secret],
    volumes={"/data": volume},
)
def run_one_query(
    query: str = DEFAULT_QUERY,
    question_id: int = 1,
    llm_backend: str = "groq",
    local_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    colbert_batch_size: int = 64,
    cross_encoder_batch_size: int = 64,
) -> dict:
    import os
    import sys
    import time

    os.chdir(REMOTE_REPO)
    sys.path.insert(0, REMOTE_REPO)

    os.environ.setdefault("LLM_BACKEND", llm_backend)
    os.environ.setdefault("LOCAL_LLM_MODEL", local_model)
    os.environ.setdefault("LOCAL_LLM_MAX_MODEL_LEN", "8192")
    os.environ.setdefault("COLBERT_BATCH_SIZE", str(colbert_batch_size))
    os.environ.setdefault("CROSS_ENCODER_BATCH_SIZE", str(cross_encoder_batch_size))
    os.environ.setdefault("RERANK_MAX_CHARS", "600")
    os.environ.setdefault("RETRIEVAL_TOP_K", "40")
    os.environ.setdefault("INITIAL_FUSION_TOP_K", "40")
    os.environ.setdefault("COLBERT_TOP_K", "20")
    os.environ.setdefault("CROSS_ENCODER_TOP_K", "10")
    os.environ.setdefault("FINAL_TOP_K", "8")
    os.environ.setdefault("QDRANT_TIMEOUT", "120")
    os.environ.setdefault("QDRANT_HNSW_EF", "64")
    os.environ.setdefault("HF_DOWNLOAD_WORKERS", "16")

    graph_path = Path("/data/indexes/graph/legal_graph.pkl")
    if graph_path.exists():
        os.environ.setdefault("GRAPH_PATH", str(graph_path))
        os.environ.setdefault("PRELOAD_GRAPH", "true")
    else:
        import pickle

        import networkx as nx

        empty_graph = Path("/tmp/legal_graph_empty.pkl")
        with empty_graph.open("wb") as file:
            pickle.dump(nx.DiGraph(), file)
        os.environ.setdefault("GRAPH_PATH", str(empty_graph))
        os.environ.setdefault("PRELOAD_GRAPH", "true")
        print(f"Không thấy graph ở {graph_path}; dùng graph rỗng để test.")

    import torch

    from src.generation.endpoint import create_llm_client
    from src.pipeline import InferencePipeline

    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else "cpu",
                "llm_backend": llm_backend,
                "local_model": local_model,
                "colbert_batch_size": colbert_batch_size,
                "cross_encoder_batch_size": cross_encoder_batch_size,
                "qdrant_url": os.getenv("QDRANT_URL"),
                "has_qdrant_api_key": bool(os.getenv("QDRANT_API_KEY")),
                "collection": os.getenv(
                    "QDRANT_COLLECTION",
                    "legal_agent_rag_harrier_idf",
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )

    started = time.perf_counter()
    llm = create_llm_client(llm_backend, local_model)
    pipeline = InferencePipeline(llm=llm, verbose=True)
    try:
        result = pipeline.run(query, question_id=question_id)
    finally:
        pipeline.close()

    total = time.perf_counter() - started
    output = {
        "total_wall_time": total,
        "latencies": result.latencies,
        "submission": result.submission.model_dump(),
        "selected_evidence": [
            {
                "unit_id": item.unit_id,
                "doc_code": item.doc_code or item.metadata.get("doc_code"),
                "article": item.article or item.metadata.get("article"),
                "final_score": item.final_score,
            }
            for item in result.selected_evidence
        ],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return output


@app.local_entrypoint()
def main(
    action: str = "run",
    query: str = DEFAULT_QUERY,
    question_id: int = 1,
    detach: bool = False,
    llm_backend: str = "groq",
    local_model: str = "Qwen/Qwen3-4B-Instruct-2507",
    colbert_batch_size: int = 64,
    cross_encoder_batch_size: int = 64,
    graph: str = "data/indexes/graph/legal_graph.pkl",
):
    if action == "upload-graph":
        with volume.batch_upload(force=True) as upload:
            upload.put_file(graph, "indexes/graph/legal_graph.pkl")
        print(f"Đã upload graph lên {VOLUME_NAME}:/indexes/graph/legal_graph.pkl")
        return
    if action != "run":
        raise ValueError("action phải là run hoặc upload-graph.")

    kwargs = {
        "query": query,
        "question_id": question_id,
        "llm_backend": llm_backend,
        "local_model": local_model,
        "colbert_batch_size": colbert_batch_size,
        "cross_encoder_batch_size": cross_encoder_batch_size,
    }
    if detach:
        call = run_one_query.spawn(**kwargs)
        print(f"Đã chạy job: {call.object_id}")
    else:
        print(json.dumps(run_one_query.remote(**kwargs), ensure_ascii=False, indent=2))
