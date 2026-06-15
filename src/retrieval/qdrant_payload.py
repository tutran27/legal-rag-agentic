from qdrant_client import QdrantClient, models

from src.common.config import settings
from src.schema.agent_schemas import Evidence


PAYLOAD_FIELDS = [
    "unit_id",
    "chunk_id",
    "chunk_type",
    "text",
    "content_text",
    "doc_id",
    "doc_code",
    "doc_type",
    "doc_title_submission",
    "article",
    "article_title",
    "domain",
    "sector",
    "status",
    "is_current",
    "part_index",
    "part_count",
    "parent_path",
]


def payload_to_evidence(
    payload: dict,
    source: str,
    score: float = 0.0,
) -> Evidence:
    return Evidence(
        unit_id=str(payload["unit_id"]),
        chunk_id=payload.get("chunk_id"),
        text=str(payload.get("text") or payload.get("content_text") or ""),
        doc_code=payload.get("doc_code"),
        doc_title_submission=payload.get("doc_title_submission"),
        article=payload.get("article"),
        article_title=payload.get("article_title"),
        source=source,
        chunk_type=payload.get("chunk_type"),
        score=score,
        final_score=score,
        metadata=payload,
    )


def scroll_payloads(
    client: QdrantClient,
    scroll_filter: models.Filter,
    limit: int,
) -> list[dict]:
    records, _ = client.scroll(
        collection_name=settings.collection_name,
        scroll_filter=scroll_filter,
        limit=limit,
        with_payload=PAYLOAD_FIELDS,
        with_vectors=False,
        timeout=settings.qdrant_timeout,
    )
    return [record.payload or {} for record in records if record.payload]
