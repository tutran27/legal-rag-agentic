from __future__ import annotations

import pickle
import re
from pathlib import Path

import pyarrow.dataset as ds

from src.schema.agent_schemas import Evidence


def graph_search(
    query: str,
    candidates: list[Evidence],
    graph_path: str = "data/indexes/graph/legal_graph.pkl",
    corpus_path: str = "data/processed/retrieval_corpus.parquet",
    top_k: int = 50,
    current_only: bool = True,
) -> list[Evidence]:
    if not candidates:
        return []

    with Path(graph_path).open("rb") as file:
        graph = pickle.load(file)

    related_docs = {}
    for candidate in candidates:
        doc_id = str(candidate.metadata.get("doc_id") or "")
        if not doc_id or doc_id not in graph:
            continue

        base_score = candidate.final_score or candidate.score or 1.0

        for _, target, data in graph.out_edges(doc_id, data=True):
            related_docs[str(target)] = {
                "score": base_score,
                "relation": data.get("relation_type", "RELATED"),
                "seed_doc_id": doc_id,
            }

        for source, _, data in graph.in_edges(doc_id, data=True):
            related_docs[str(source)] = {
                "score": base_score,
                "relation": data.get("relation_type", "RELATED"),
                "seed_doc_id": doc_id,
            }

    if not related_docs:
        return []

    dataset = ds.dataset(corpus_path, format="parquet")
    condition = ds.field("doc_id").isin(list(related_docs))
    if current_only:
        condition &= ds.field("is_current") == True

    rows = dataset.to_table(filter=condition).to_pylist()
    if not rows:
        return []

    query_tokens = set(re.findall(r"\w+", query.lower()))
    scores = [
        len(query_tokens & set(re.findall(r"\w+", row["text"].lower())))
        for row in rows
    ]

    results = []
    for row, text_score in zip(rows, scores):
        graph_info = related_docs[row["doc_id"]]
        score = float(text_score) + graph_info["score"]
        results.append(
            Evidence(
                unit_id=row["unit_id"],
                chunk_id=row["chunk_id"],
                text=row["text"],
                doc_code=row["doc_code"],
                doc_title_submission=row["doc_title_submission"],
                article=row["article"],
                article_title=row["article_title"],
                source="graph",
                chunk_type=row["chunk_type"],
                score=score,
                final_score=score,
                metadata={
                    "doc_id": row["doc_id"],
                    "relation_type": graph_info["relation"],
                    "seed_doc_id": graph_info["seed_doc_id"],
                    "status": row["status"],
                    "is_current": row["is_current"],
                },
            )
        )

    results.sort(key=lambda item: item.final_score, reverse=True)
    unique_results = {}
    for result in results:
        unique_results.setdefault(result.unit_id, result)
    return list(unique_results.values())[:top_k]
