from __future__ import annotations

import pyarrow.dataset as ds
import re

from src.schema.agent_schemas import Evidence


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def expand_context(
    candidates: list[Evidence],
    query: str = "",
    corpus_path: str = "data/processed/retrieval_corpus.parquet",
    top_k: int = 50,
) -> list[Evidence]:
    unit_ids = list({candidate.unit_id for candidate in candidates})
    if not unit_ids:
        return []

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
        if distance != 1:
            continue

        query_tokens = _tokens(query)
        row_tokens = _tokens(row.get("content_text") or row["text"])
        overlap = len(query_tokens & row_tokens) / max(len(query_tokens), 1)
        score = 0.5 * float(seed_score) + overlap
        results.append(
            Evidence(
                unit_id=row["unit_id"],
                chunk_id=row["chunk_id"],
                text=row["text"],
                source="context",
                score=score,
                final_score=score,
                metadata=row,
            )
        )
    results.sort(key=lambda item: item.final_score, reverse=True)
    return results[:top_k]
