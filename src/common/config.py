import os

from dotenv import load_dotenv


load_dotenv()


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


def _get_str(name: str, default: str) -> str:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().strip("'").strip('"')


class Settings:
    llm_backend = _get_str("LLM_BACKEND", "endpoint")
    llm_endpoint_url = _get_str(
        "LLM_ENDPOINT_URL",
        "https://hieudan2810--qwen2-5-14b-vllm-serve.modal.run",
    )
    llm_endpoint_model = _get_str(
        "LLM_ENDPOINT_MODEL",
        "qwen2.5-14b-instruct",
    )
    llm_endpoint_timeout = _get_int("LLM_ENDPOINT_TIMEOUT", 600)
    qwen_api_key = os.getenv("QWEN_API_KEY")

    groq_model = _get_str("GROQ_MODEL", "llama-3.1-8b-instant")
    groq_base_url = _get_str(
        "GROQ_BASE_URL",
        "https://api.groq.com/openai/v1",
    )
    groq_timeout = _get_int("GROQ_TIMEOUT", 120)
    groq_retry_attempts = _get_int("GROQ_RETRY_ATTEMPTS", 4)
    groq_retry_delay = _get_int("GROQ_RETRY_DELAY", 5)
    groq_api_key = os.getenv("GROQ_API_KEY")
    groq_api_keys = os.getenv("GROQ_API_KEYS")

    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    openrouter_model = _get_str(
        "OPENROUTER_MODEL",
        "meta-llama/llama-3.1-8b-instruct",
    )
    openrouter_base_url = _get_str(
        "OPENROUTER_BASE_URL",
        "https://openrouter.ai/api/v1",
    )
    openrouter_timeout = _get_int("OPENROUTER_TIMEOUT", 120)
    openrouter_retry_attempts = _get_int("OPENROUTER_RETRY_ATTEMPTS", 4)
    openrouter_retry_delay = _get_int("OPENROUTER_RETRY_DELAY", 5)

    local_llm_model = _get_str(
        "LOCAL_LLM_MODEL",
        "Qwen/Qwen3-4B-Instruct-2507",
    )
    local_llm_max_model_len = _get_int("LOCAL_LLM_MAX_MODEL_LEN", 0)
    local_llm_load_in_4bit = _get_bool("LOCAL_LLM_LOAD_IN_4BIT", True)

    qdrant_url = _get_str("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY")
    collection_name = _get_str(
        "QDRANT_COLLECTION",
        "legal_agent_rag_harrier_idf",
    )

    top_n = _get_int("TOP_N", 50)
    top_k = _get_int("TOP_K", 10)
    batch_size = _get_int("BATCH_SIZE", 128)
    normalize_dense = _get_bool("NORMALIZE_DENSE", True)

    dense_model = _get_str(
        "DENSE_MODEL", "mainguyen9/vietlegal-harrier-0.6b"
    )
    colbert_model = _get_str(
        "COLBERT_MODEL",
        _get_str("BGE_MODEL", "BAAI/bge-m3"),
    )
    dense_dim = _get_int("DENSE_DIM", 1024)

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    hf_download_workers = _get_int("HF_DOWNLOAD_WORKERS", 8)

    qdrant_upsert_batch_size = _get_int("QDRANT_UPSERT_BATCH_SIZE", 64)
    qdrant_upload_workers = _get_int("QDRANT_UPLOAD_WORKERS", 4)
    qdrant_timeout = _get_int("QDRANT_TIMEOUT", 120)
    qdrant_hnsw_ef = _get_int("QDRANT_HNSW_EF", 64)
    colbert_batch_size = _get_int("COLBERT_BATCH_SIZE", 16)
    cross_encoder_batch_size = _get_int("CROSS_ENCODER_BATCH_SIZE", 16)
    retrieval_top_k = _get_int("RETRIEVAL_TOP_K", 30)
    initial_fusion_top_k = _get_int("INITIAL_FUSION_TOP_K", 30)
    colbert_top_k = _get_int("COLBERT_TOP_K", 15)
    cross_encoder_top_k = _get_int("CROSS_ENCODER_TOP_K", 8)
    graph_seed_top_k = _get_int("GRAPH_SEED_TOP_K", 5)
    graph_top_k = _get_int("GRAPH_TOP_K", 10)
    context_top_k = _get_int("CONTEXT_TOP_K", 10)
    final_top_k = _get_int("FINAL_TOP_K", 6)
    rerank_max_chars = _get_int("RERANK_MAX_CHARS", 1000)
    rerank_device = _get_str("RERANK_DEVICE", "cuda")
    cross_encoder_model = _get_str(
        "CROSS_ENCODER_MODEL",
        "Qwen/Qwen3-Reranker-0.6B",
    )
    graph_path = _get_str(
        "GRAPH_PATH",
        "data/indexes/graph/legal_graph.pkl",
    )
    preload_graph = _get_bool("PRELOAD_GRAPH", True)
    store_text = _get_bool("STORE_TEXT", True)
    enable_colbert = _get_bool("ENABLE_COLBERT", True)
    enable_cross_encoder = _get_bool("ENABLE_CROSS_ENCODER", True)
    enable_reasoning = _get_bool("ENABLE_REASONING", True)
    retrieval_only = _get_bool("RETRIEVAL_ONLY", False)


settings = Settings()
