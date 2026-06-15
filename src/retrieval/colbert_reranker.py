from __future__ import annotations

import numpy as np
import torch

from src.common.config import settings
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
    batch_size: int = settings.colbert_batch_size,
) -> list[Evidence]:
    if not candidates:
        return []

    documents = [
        (
            f"{candidate.metadata.get('article_title') or ''}\n"
            f"{candidate.metadata.get('content_text') or candidate.text}"
        )[:settings.rerank_max_chars]
        for candidate in candidates
    ]
    with torch.inference_mode():
        embeddings = colbert_model.encode(
            [query, *documents],
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
