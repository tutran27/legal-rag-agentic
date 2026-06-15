from __future__ import annotations

from collections import defaultdict

from src.schema.agent_schemas import Evidence


def rrf_fusion(
    result_sets: list[list[Evidence]],
    top_k: int = 100,
    rrf_k: int = 60,
    weights: list[float] | None = None,
) -> list[Evidence]:
    if weights is None:
        weights = [1.0] * len(result_sets)
    if len(weights) != len(result_sets):
        raise ValueError("Số weights phải bằng số result_sets.")

    candidates: dict[str, Evidence] = {}
    scores: defaultdict[str, float] = defaultdict(float)
    votes: defaultdict[str, int] = defaultdict(int)

    for results, weight in zip(result_sets, weights):
        for rank, candidate in enumerate(results, start=1):
            key = candidate.chunk_id or candidate.unit_id
            if not key:
                continue
            candidates.setdefault(key, candidate)
            scores[key] += weight / (rrf_k + rank)
            votes[key] += 1

    fused = [
        candidate.model_copy(
            update={
                "score": scores[key],
                "final_score": scores[key],
                "vote_count": votes[key],
            }
        )
        for key, candidate in candidates.items()
    ]
    return sorted(
        fused,
        key=lambda candidate: candidate.final_score,
        reverse=True,
    )[:top_k]
