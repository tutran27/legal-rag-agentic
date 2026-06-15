from sentence_transformers import CrossEncoder
import numpy as np
import torch

from src.common.config import settings

model_name="Qwen/Qwen3-Reranker-0.6B"


def _minmax(values: list[float]) -> list[float]:
    low = min(values)
    high = max(values)
    value_range = high - low
    if not value_range:
        return [1.0] * len(values)
    return [(value - low) / value_range for value in values]


def cross_encoder_rerank(
    query: str,
    candidates,
    model: CrossEncoder = None,
    model_name: str = model_name,
    top_k: int = 30,
    batch_size: int = settings.cross_encoder_batch_size,
):
    if not candidates:
        return []

    if model is None:
        model = CrossEncoder(model_name)

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
    raw_scores = [float(score) for score in scores]
    ce_scores = _minmax(raw_scores)
    colbert_scores = [
        candidate.colbert_normalized_score or 0.0
        for candidate in candidates
    ]
    fusion_scores = _minmax(
        [candidate.final_score for candidate in candidates]
    )
    reranked = [
        candidate.model_copy(
            update={
                "cross_encoder_rerank_score": raw_score,
                "cross_encoder_normalized_score": ce_score,
                "final_score": (
                    0.7 * ce_score
                    + 0.2 * colbert_score
                    + 0.1 * fusion_score
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
