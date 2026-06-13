from sentence_transformers import CrossEncoder
from src.common.config import settings

model_name="Qwen/Qwen3-Reranker-0.6B"

def cross_encoder_rerank(
    query: str,
    candidates,
    model: CrossEncoder = None,
    model_name: str = model_name,
    top_k: int = 30,
    batch_size: int = 10,
):
    if not candidates:
        return []

    if model is None:
        model = CrossEncoder(model_name)

    scores = model.predict(
        [(query, candidate.text) for candidate in candidates],
        batch_size=batch_size,
    )
    reranked = [
        candidate.model_copy(
            update={"rerank_score": float(score), "final_score": float(score)}
        )
        for candidate, score in zip(candidates, scores)
    ]
    reranked.sort(key=lambda candidate: candidate.rerank_score or 0.0, reverse=True)
    return reranked[:top_k]
