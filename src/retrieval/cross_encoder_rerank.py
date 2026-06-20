from sentence_transformers import CrossEncoder
import numpy as np
import torch

from src.common.config import settings
from src.common.embedding import get_torch_device

model_name = settings.cross_encoder_model


def _minmax(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    value_range = high - low
    if not value_range:
        return [1.0 if high > 0 else 0.0] * len(values)
    return [(value - low) / value_range for value in values]


def cross_encoder_rerank(
    query: str,
    candidates,
    model: CrossEncoder = None,
    model_name: str = model_name,
    top_k: int = settings.final_top_k,
    batch_size: int = settings.cross_encoder_batch_size,
):
    if not candidates:
        return []

    if model is None:
        device = get_torch_device()
        model = CrossEncoder(
            model_name,
            device=device,
            model_kwargs={"torch_dtype": torch.float16}
            if device == "cuda"
            else None,
        )

    pairs = [
        (
            query,
            (
                f"{candidate.metadata.get('article_title') or ''}\n"
                f"{candidate.metadata.get('content_text') or candidate.text}"
            )[:settings.rerank_max_chars],
        )
        for candidate in candidates
    ]
    with torch.inference_mode():
        scores = np.asarray(
            model.predict(pairs, batch_size=batch_size)
        ).reshape(-1)
    raw_scores = [max(0.0, float(score)) for score in scores]
    ce_scores = _minmax(raw_scores)
    use_colbert = any(
        candidate.colbert_normalized_score is not None
        for candidate in candidates
    )
    colbert_scores = [
        candidate.colbert_normalized_score or 0.0 for candidate in candidates
    ]
    fusion_scores = _minmax(
        [candidate.final_score for candidate in candidates]
    )
    weights = (0.4, 0.4, 0.2) if use_colbert else (0.7, 0.0, 0.3)
    reranked = [
        candidate.model_copy(
            update={
                "cross_encoder_rerank_score": raw_score,
                "cross_encoder_normalized_score": ce_score,
                "final_score": (
                    weights[0] * ce_score
                    + weights[1] * colbert_score
                    + weights[2] * fusion_score
                ),
            }
        )
        for candidate, raw_score, ce_score, colbert_score, fusion_score in zip(
            candidates,
            raw_scores,
            ce_scores,
            colbert_scores,
            fusion_scores,
        )
    ]
    reranked.sort(
        key=lambda candidate: candidate.final_score,
        reverse=True,
    )
    return reranked[:top_k]
