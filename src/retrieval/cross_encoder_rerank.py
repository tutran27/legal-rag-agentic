from sentence_transformers import CrossEncoder

model_name="Qwen/Qwen3-Reranker-0.6B"

def cross_encoder_rerank(
    query: str,
    candidates,
    model: CrossEncoder,
    top_k: int = 20,
    batch_size: int = 10):
    if not candidates:
        return []
    pairs=[(query, candidate.payload["text"]) for candidate in candidates]
    scores=model.predict(pairs, batch_size=batch_size)
    for i, score in enumerate(scores):
        candidates[i].payload["cross_encoder_score"]=float(score)
    candidates.sort(key=lambda x: x.payload["cross_encoder_score"], reverse=True)
    return candidates[:top_k]
