from __future__ import annotations

import pyarrow.dataset as ds
import re
from qdrant_client import QdrantClient, models

from src.retrieval.qdrant_payload import payload_to_evidence, scroll_payloads
from src.schema.agent_schemas import Evidence


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _distance_boost(distance: int) -> float:
    return 1.0 / (1.0 + float(distance))


def expand_context(
    candidates: list[Evidence],
    query: str = "",
    corpus_path: str = "data/processed/retrieval_corpus.parquet",
    top_k: int = 50,
    client: QdrantClient | None = None,
) -> list[Evidence]:
    unit_ids = list({candidate.unit_id for candidate in candidates})
    if not unit_ids:
        return []

    if client is not None:
        rows = scroll_payloads(
            client,
            models.Filter(
                must=[
                    models.FieldCondition(
                        key="unit_id",
                        match=models.MatchAny(any=unit_ids),
                    )
                ]
            ),
            limit=max(100, len(unit_ids) * 20),
        )
    else:
        dataset = ds.dataset(corpus_path, format="parquet")
        rows = dataset.to_table(
            filter=ds.field("unit_id").isin(unit_ids)
        ).to_pylist()
    seed_chunks = {
        candidate.chunk_id for candidate in candidates if candidate.chunk_id
    }
    seeds_by_unit = {}
    for candidate in candidates:
        part_index = candidate.metadata.get("part_index")
        if part_index is None:
            continue
        seeds_by_unit.setdefault(candidate.unit_id, []).append(
            (int(part_index), candidate.final_score or candidate.score)
        )

    results = []
    for row in rows:
        if row["chunk_id"] in seed_chunks:
            continue
        part_index = row.get("part_index")
        unit_seeds = seeds_by_unit.get(row["unit_id"], [])
        if part_index is None or not unit_seeds:
            continue

        distance, seed_score = min(
            (
                abs(int(part_index) - seed_index),
                score,
            )
            for seed_index, score in unit_seeds
        )
        query_tokens = _tokens(query)
        row_tokens = _tokens(row.get("content_text") or row["text"])
        overlap = len(query_tokens & row_tokens) / max(len(query_tokens), 1)
        score = (
            0.45 * float(seed_score)
            + 0.35 * _distance_boost(distance)
            + 0.20 * overlap
        )
        results.append(
            payload_to_evidence(row, source="context", score=score)
        )
    results.sort(key=lambda item: item.final_score, reverse=True)
    return results[:top_k]
