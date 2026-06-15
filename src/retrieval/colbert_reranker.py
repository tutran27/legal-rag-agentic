from __future__ import annotations

import numpy as np

from src.schema.agent_schemas import Evidence


def colbert_score(query_vectors, document_vectors) -> float:
    query_vectors = np.asarray(query_vectors, dtype=np.float32)
    document_vectors = np.asarray(document_vectors, dtype=np.float32)
    scores = query_vectors @ document_vectors.T
    return float(scores.max(axis=1).mean())


def colbert_rerank(
    query: str,
    candidates,
    colbert_model,
    top_k: int = 50,
    batch_size: int = 8,
) -> list[Evidence]:
    if not candidates:
        return []

    embeddings = colbert_model.encode(
        [query, *[candidate.text for candidate in candidates]],
        batch_size=batch_size,
        return_dense=False,
        return_sparse=False,
        return_colbert_vecs=True,
    )["colbert_vecs"]

    query_vectors = embeddings[0]
    scored = [
        candidate.model_copy(
            update={
                "colbert_rerank_score": colbert_score(
                    query_vectors,
                    document_vectors,
                )
            }
        )
        for candidate, document_vectors in zip(candidates, embeddings[1:])
    ]

    values = [candidate.colbert_rerank_score for candidate in scored]
    min_score = min(values)
    max_score = max(values)
    score_range = max_score - min_score
    reranked = [
        candidate.model_copy(
            update={
                "colbert_normalized_score": (
                    (candidate.colbert_rerank_score - min_score) / score_range
                    if score_range
                    else 1.0
                )
            }
        )
        for candidate in scored
    ]
    reranked.sort(
        key=lambda candidate: candidate.colbert_rerank_score or 0.0,
        reverse=True,
    )
    return reranked[:top_k]
