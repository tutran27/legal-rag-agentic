from __future__ import annotations

import pickle
import re
from functools import lru_cache
from pathlib import Path

import pyarrow.dataset as ds
from qdrant_client import QdrantClient, models

from src.common.bm25 import bm25_vector
from src.common.config import settings
from src.retrieval.qdrant_payload import PAYLOAD_FIELDS, payload_to_evidence
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


@lru_cache(maxsize=1)
def load_graph(graph_path: str):
    with Path(graph_path).open("rb") as file:
        return pickle.load(file)


def graph_search(
    query: str,
    candidates: list[Evidence],
    graph_path: str = "data/indexes/graph/legal_graph.pkl",
    corpus_path: str = "data/processed/retrieval_corpus.parquet",
    top_k: int = 50,
    current_only: bool = True,
    max_related_docs: int = 30,
    client: QdrantClient | None = None,
) -> list[Evidence]:
    if not candidates:
        return []

    graph = load_graph(graph_path)

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
    if client is not None:
        conditions = [
            models.FieldCondition(
                key="doc_id",
                match=models.MatchAny(any=list(related_docs)),
            )
        ]
        if current_only:
            conditions.append(
                models.FieldCondition(
                    key="is_current",
                    match=models.MatchValue(value=True),
                )
            )
        response = client.query_points(
            collection_name=settings.collection_name,
            query=bm25_vector(query),
            using="sparse",
            query_filter=models.Filter(must=conditions),
            limit=max(top_k * 3, 30),
            with_payload=PAYLOAD_FIELDS,
            with_vectors=False,
            timeout=settings.qdrant_timeout,
        )
        rows = [
            (point.payload or {}, float(point.score))
            for point in response.points
            if point.payload
        ]
    else:
        dataset = ds.dataset(corpus_path, format="parquet")
        condition = ds.field("doc_id").isin(list(related_docs))
        if current_only:
            condition &= ds.field("is_current") == True
        rows = [
            (row, None)
            for row in dataset.to_table(filter=condition).to_pylist()
        ]
    if not rows:
        return []

    query_tokens = _tokens(query)

    results = []
    for row, retrieval_score in rows:
        graph_info = related_docs[row["doc_id"]]
        text = " ".join(
            [
                str(row.get("article_title") or ""),
                str(row.get("content_text") or row.get("text") or ""),
            ]
        )
        text_tokens = _tokens(text)
        overlap = (
            retrieval_score
            if retrieval_score is not None
            else len(query_tokens & text_tokens) / max(len(query_tokens), 1)
        )
        score = graph_info["score"] + overlap
        metadata = {
            **row,
            "relation_type": graph_info["relation"],
            "seed_doc_id": graph_info["seed_doc_id"],
        }
        results.append(
            payload_to_evidence(metadata, source="graph", score=score)
        )

    results.sort(key=lambda item: item.final_score, reverse=True)
    unique_results = {}
    for result in results:
        unique_results.setdefault(result.unit_id, result)
    return list(unique_results.values())[:top_k]
