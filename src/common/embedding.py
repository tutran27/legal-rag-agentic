import os

from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from FlagEmbedding import BGEM3FlagModel

from src.common.config import Settings
    
dense_model = Settings.dense_model
sparse_model = Settings.bge_model


def download_model(model_name: str, hf_token: str | None = None) -> str:
    return snapshot_download(
        repo_id=model_name,
        token=hf_token,
        resume_download=True,
        max_workers=1,
    )


def load_model(
    dense_model: str = dense_model,
    sparse_model: str = sparse_model,
    hf_token: str | None = None,
):
    hf_token = hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")

    print(f"Downloading dense model: {dense_model}")
    dense_model_path = download_model(dense_model, hf_token=hf_token)
    print(f"Loading dense model from: {dense_model_path}")
    dense_st = SentenceTransformer(dense_model_path, token=hf_token)

    print(f"Downloading BGE-M3 model: {sparse_model}")
    sparse_model_path = download_model(sparse_model, hf_token=hf_token)
    print(f"Loading BGE-M3 model from: {sparse_model_path}")
    bge_m3 = BGEM3FlagModel(sparse_model_path, use_fp16=True)
    return dense_st, bge_m3


def embed_text(
    text: list[str],
    dense_model: SentenceTransformer,
    bge_model: BGEM3FlagModel,
    batch_size: int = 32,
    normalize_dense: bool = True,
):
    dense_embedding = dense_model.encode(
        text,
        normalize_embeddings=normalize_dense,
        batch_size=batch_size,
        show_progress_bar=False,
    )
    bge_embedding = bge_model.encode(
        text,
        batch_size=batch_size,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=True,
    )

    sparse_embedding = bge_embedding["sparse_vecs"]
    colbert_embedding = bge_embedding["colbert_vecs"]
    return dense_embedding, sparse_embedding, colbert_embedding
