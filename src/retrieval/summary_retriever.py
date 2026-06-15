from __future__ import annotations

import re

import pyarrow.dataset as ds
import pyarrow.parquet as pq

from src.schema.agent_schemas import Evidence


STOPWORDS = {
    "các", "có", "của", "được", "là", "những", "theo", "trong", "và", "về",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"\w+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def summary_search(
    query: str,
    documents_path: str = "data/processed/documents.parquet",
    corpus_path: str = "data/processed/retrieval_corpus.parquet",
    top_k: int = 30,
    current_only: bool = True,
) -> list[Evidence]:
    normalized_query = " ".join(re.findall(r"\w+", query.lower()))
    query_tokens = _tokens(query)
    documents = pq.read_table(
        documents_path,
        columns=[
            "doc_id",
            "doc_title_submission",
            "domain",
            "sector",
            "scope",
            "status",
        ],
    ).to_pylist()

    ranked_docs = []
    for document in documents:
        summary = " ".join(
            str(document.get(field) or "")
            for field in (
                "doc_title_submission",
                "domain",
                "sector",
                "scope",
            )
        )
        normalized_summary = " ".join(re.findall(r"\w+", summary.lower()))
        score = len(query_tokens & _tokens(normalized_summary))
        if normalized_query in normalized_summary:
            score += 100
        status = str(document["status"] or "").strip().lower()
        expired = status in {"hết hiệu lực", "hết hiệu lực toàn bộ"}
        if score and (not current_only or not expired):
            ranked_docs.append(
                (score, -len(normalized_summary), str(document["doc_id"]))
            )

    ranked_docs.sort(reverse=True)
    doc_ids = [
        doc_id
        for _, _, doc_id in ranked_docs[: min(top_k, 10)]
        if doc_id
    ]
    if not doc_ids:
        return []

    dataset = ds.dataset(corpus_path, format="parquet")
    condition = ds.field("doc_id").isin(doc_ids)
    if current_only:
        condition &= ds.field("is_current") == True
    rows = dataset.to_table(filter=condition).to_pylist()

    ranked_rows = []
    for row in rows:
        text = " ".join(
            [
                str(row.get("article_title") or ""),
                str(row.get("content_text") or row.get("text") or ""),
            ]
        )
        relevance = len(query_tokens & _tokens(text))
        if relevance:
            ranked_rows.append((relevance, row))

    ranked_rows.sort(key=lambda item: item[0], reverse=True)
    results = []
    per_doc = {}
    for relevance, row in ranked_rows:
        if per_doc.get(row["doc_id"], 0) >= 2:
            continue
        per_doc[row["doc_id"]] = per_doc.get(row["doc_id"], 0) + 1
        results.append(
            Evidence(
                unit_id=row["unit_id"],
                chunk_id=row["chunk_id"],
                text=row["text"],
                source="summary",
                score=float(relevance),
                final_score=float(relevance),
                metadata=row,
            )
        )
        if len(results) >= top_k:
            break
    return results[:top_k]
