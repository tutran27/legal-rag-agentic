from __future__ import annotations

import pickle
import re
from pathlib import Path

import pyarrow.dataset as ds

from src.schema.agent_schemas import Evidence


RELATION_WEIGHTS = {
    "AMENDS": 1.0,
    "GUIDES": 0.9,
    "REFERENCES": 0.6,
    "RELATED": 0.25,
}
STOPWORDS = {
    "các", "có", "của", "được", "là", "những", "theo", "trong", "và", "về",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def graph_search(
    query: str,
    candidates: list[Evidence],
    graph_path: str = "data/indexes/graph/legal_graph.pkl",
    corpus_path: str = "data/processed/retrieval_corpus.parquet",
    top_k: int = 50,
    current_only: bool = True,
    max_related_docs: int = 30,
) -> list[Evidence]:
    if not candidates:
        return []

    with Path(graph_path).open("rb") as file:
        graph = pickle.load(file)

    related_docs = {}
    seed_doc_ids = {
        str(candidate.metadata.get("doc_id") or "")
        for candidate in candidates
    }

    def add_related(doc_id: str, relation: str, seed_doc_id: str, score: float):
        if not doc_id or doc_id in seed_doc_ids:
            return
        current = related_docs.get(doc_id)
        if current is None or score > current["score"]:
            related_docs[doc_id] = {
                "score": score,
                "relation": relation,
                "seed_doc_id": seed_doc_id,
            }

    for candidate in candidates:
        doc_id = str(candidate.metadata.get("doc_id") or "")
        if not doc_id or doc_id not in graph:
            continue

        base_score = candidate.final_score or candidate.score or 1.0

        for _, target, data in graph.out_edges(doc_id, data=True):
            relation = str(data.get("relation_type", "RELATED")).upper()
            add_related(
                str(target),
                relation,
                doc_id,
                base_score + RELATION_WEIGHTS.get(relation, 0.2),
            )

        for source, _, data in graph.in_edges(doc_id, data=True):
            relation = str(data.get("relation_type", "RELATED")).upper()
            add_related(
                str(source),
                relation,
                doc_id,
                base_score + RELATION_WEIGHTS.get(relation, 0.2),
            )

    if not related_docs:
        return []

    related_docs = dict(
        sorted(
            related_docs.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )[:max_related_docs]
    )
    dataset = ds.dataset(corpus_path, format="parquet")
    condition = ds.field("doc_id").isin(list(related_docs))
    if current_only:
        condition &= ds.field("is_current") == True

    rows = dataset.to_table(filter=condition).to_pylist()
    if not rows:
        return []

    query_tokens = _tokens(query)

    results = []
    for row in rows:
        graph_info = related_docs[row["doc_id"]]
        text = " ".join(
            [
                str(row.get("article_title") or ""),
                str(row.get("content_text") or row.get("text") or ""),
            ]
        )
        text_tokens = _tokens(text)
        overlap = len(query_tokens & text_tokens) / max(len(query_tokens), 1)
        score = graph_info["score"] + overlap
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
                    **row,
                    "relation_type": graph_info["relation"],
                    "seed_doc_id": graph_info["seed_doc_id"],
                },
            )
        )

    results.sort(key=lambda item: item.final_score, reverse=True)
    unique_results = {}
    for result in results:
        unique_results.setdefault(result.unit_id, result)
    return list(unique_results.values())[:top_k]
