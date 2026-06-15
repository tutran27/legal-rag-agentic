from __future__ import annotations

import re

import pyarrow.dataset as ds
from qdrant_client import QdrantClient, models

from src.retrieval.qdrant_payload import payload_to_evidence, scroll_payloads
from src.schema.agent_schemas import Evidence


DOC_CODE_RE = re.compile(
    r"\b\d+(?:-\w+)?/\d{4}/[\wĐĐ-]+\b",
    re.IGNORECASE,
)
ARTICLE_RE = re.compile(r"\bđiều\s+(\d+[a-z]?)\b", re.IGNORECASE)


def exact_search(
    query: str,
    doc_codes: list[str] | None = None,
    corpus_path: str = "data/processed/retrieval_corpus.parquet",
    top_k: int = 20,
    current_only: bool = True,
    client: QdrantClient | None = None,
) -> list[Evidence]:
    codes = doc_codes or DOC_CODE_RE.findall(query)
    article_match = ARTICLE_RE.search(query)
    if not codes and not article_match:
        return []

    if client is not None:
        conditions = []
        if codes:
            conditions.append(
                models.FieldCondition(
                    key="doc_code",
                    match=models.MatchAny(any=codes),
                )
            )
        if article_match:
            conditions.append(
                models.FieldCondition(
                    key="article",
                    match=models.MatchValue(
                        value=f"Điều {article_match.group(1)}"
                    ),
                )
            )
        if current_only:
            conditions.append(
                models.FieldCondition(
                    key="is_current",
                    match=models.MatchValue(value=True),
                )
            )
        payloads = scroll_payloads(
            client,
            models.Filter(must=conditions),
            limit=top_k,
        )
        return [
            payload_to_evidence(payload, source="exact", score=1.0)
            for payload in payloads
        ]

    dataset = ds.dataset(corpus_path, format="parquet")
    condition = None
    if codes:
        condition = ds.field("doc_code").isin(codes)
    if article_match:
        article = f"Điều {article_match.group(1)}"
        article_condition = ds.field("article") == article
        condition = (
            article_condition
            if condition is None
            else condition & article_condition
        )

    rows = dataset.to_table(filter=condition).slice(0, top_k).to_pylist()
    return [
        Evidence(
            unit_id=row["unit_id"],
            chunk_id=row["chunk_id"],
            text=row["text"],
            source="exact",
            score=1.0,
            metadata=row,
        )
        for row in rows
    ]
