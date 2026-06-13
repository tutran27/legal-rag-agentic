from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer

from src.common.config import settings


def download_model(model_name: str, hf_token: str | None = None) -> str:
    return snapshot_download(
        repo_id=model_name,
        token=hf_token,
        max_workers=settings.hf_download_workers,
    )


def load_dense_model(
    model_name: str = settings.dense_model,
    hf_token: str | None = None,
) -> SentenceTransformer:
    token = hf_token or settings.hf_token
    model_path = download_model(model_name, token)
    return SentenceTransformer(model_path, token=token)


def load_colbert_model(
    model_name: str = settings.colbert_model,
    hf_token: str | None = None,
) -> BGEM3FlagModel:
    model_path = download_model(model_name, hf_token or settings.hf_token)
    return BGEM3FlagModel(model_path, use_fp16=True)


def embed_dense(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = settings.batch_size,
):
    return model.encode(
        texts,
        normalize_embeddings=settings.normalize_dense,
        batch_size=batch_size,
        show_progress_bar=False,
    )
