from FlagEmbedding import BGEM3FlagModel
from huggingface_hub import snapshot_download
from sentence_transformers import SentenceTransformer
import torch

from src.common.config import settings


HARRIER_QUERY_INSTRUCTION = (
    "Instruct: Given a Vietnamese legal question, retrieve relevant legal "
    "passages that answer the question\nQuery: "
)


def get_torch_device() -> str:
    configured = settings.rerank_device.lower()
    if configured == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "RERANK_DEVICE=cuda nhưng PyTorch hiện tại không hỗ trợ CUDA. "
                "Cài PyTorch bản CUDA hoặc đổi RERANK_DEVICE=cpu/auto."
            )
        return "cuda"
    if configured == "cpu":
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


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
    device = get_torch_device()
    model_kwargs = (
        {"torch_dtype": torch.float16}
        if device == "cuda"
        else None
    )
    return SentenceTransformer(
        model_path,
        token=token,
        device=device,
        model_kwargs=model_kwargs,
    )


def load_colbert_model(
    model_name: str = settings.colbert_model,
    hf_token: str | None = None,
) -> BGEM3FlagModel:
    model_path = download_model(model_name, hf_token or settings.hf_token)
    device = get_torch_device()
    return BGEM3FlagModel(
        model_path,
        use_fp16=device == "cuda",
        devices=device,
    )


def embed_dense(
    texts: list[str],
    model: SentenceTransformer,
    batch_size: int = settings.batch_size,
    is_query: bool = False,
):
    if is_query:
        texts = [
            f"{HARRIER_QUERY_INSTRUCTION}{text}"
            for text in texts
        ]
    return model.encode(
        texts,
        normalize_embeddings=settings.normalize_dense,
        batch_size=batch_size,
        show_progress_bar=False,
    )
