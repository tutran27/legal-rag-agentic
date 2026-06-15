
import os


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


class Settings:
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name = os.getenv(
        "QDRANT_COLLECTION", "legal_agent_rag_harrier_idf"
    )

    top_n = _get_int("TOP_N", 50)
    top_k = _get_int("TOP_K", 10)
    batch_size = _get_int("BATCH_SIZE", 128)
    normalize_dense = _get_bool("NORMALIZE_DENSE", True)

    dense_model = os.getenv(
        "DENSE_MODEL", "mainguyen9/vietlegal-harrier-0.6b"
    )
    colbert_model = os.getenv(
        "COLBERT_MODEL",
        os.getenv("BGE_MODEL", "BAAI/bge-m3"),
    )
    dense_dim = _get_int("DENSE_DIM", 1024)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    hf_download_workers = _get_int("HF_DOWNLOAD_WORKERS", 8)

    qdrant_upsert_batch_size = _get_int("QDRANT_UPSERT_BATCH_SIZE", 64)
    qdrant_upload_workers = _get_int("QDRANT_UPLOAD_WORKERS", 4)
    qdrant_timeout = _get_int("QDRANT_TIMEOUT", 120)
    store_text = _get_bool("STORE_TEXT", True)


settings = Settings()
