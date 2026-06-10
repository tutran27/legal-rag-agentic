from __future__ import annotations
import numpy as np
from src.schema.agent_schemas import Evidence

def colbert_score(query_vectors, document_vectors) -> float:
    query_vectors=np.asarray(query_vectors, dtype=np.float32)
    document_vectors=np.asarray(document_vectors, dtype=np.float32)
    scores = query_vectors @ document_vectors.T
    return float(scores.max(axis=1).mean())

def colbert_rerank(
    query: str,
    candidates,
    bge_model,
    top_k: int = 20,
    batch_size: int = 16,
) -> list[Evidence]:
    if not candidates:
        return []

    texts = [candidate.text for candidate in candidates]
    embeddings = bge_model.encode(
        [query, *texts],
        batch_size=batch_size,
        return_dense=False,
        return_sparse=False,
        return_colbert_vecs=True,
    )["colbert_vecs"]

    query_vectors = embeddings[0]
    reranked = []

    for candidate, document_vectors in zip(candidates, embeddings[1:]):
        score = colbert_score(query_vectors, document_vectors)
        reranked.append(
            candidate.model_copy(
                update={
                    "rerank_score": score,
                    "final_score": score,
                }
            )
        )

    reranked.sort(key=lambda candidate: candidate.rerank_score or 0.0, reverse=True)
    return reranked[:top_k]
