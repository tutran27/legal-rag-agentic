from __future__ import annotations

import os
import re
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from src.chunking.article_chunk import build_article_chunk


CLAUSE_BOUNDARY_RE = re.compile(
    r"(?=(?:^|\s)(?:\d+\.\s|[a-zđ]\)\s))",
    flags=re.IGNORECASE,
)

RETRIEVAL_STRING_COLUMNS = [
    "chunk_id",
    "unit_id",
    "chunk_type",
    "text",
    "content_text",
    "article_id",
    "doc_id",
    "doc_code",
    "doc_type",
    "doc_title_submission",
    "doc_name_for_submission",
    "article",
    "article_title",
    "parent_path",
    "domain",
    "sector",
    "status",
    "effective_from",
    "effective_to",
    "content_source",
    "source_dataset",
    "source_url",
]
RETRIEVAL_SCHEMA = pa.schema(
    [(column, pa.string()) for column in RETRIEVAL_STRING_COLUMNS]
    + [
        ("part_index", pa.int32()),
        ("part_count", pa.int32()),
        ("is_current", pa.bool_()),
    ]
)


def split_article_text(
    text: str,
    max_chars: int = 1800,
) -> list[str]:
    text = str(text or "").strip()
    if not text:
        return []
    if len(text) <= max_chars:
        return [text]

    parts = [line.strip() for line in text.splitlines() if line.strip()]
    if len(parts) == 1:
        parts = [
            part.strip()
            for part in CLAUSE_BOUNDARY_RE.split(text)
            if part.strip()
        ]

    segments = [
        segment
        for part in parts
        for segment in (
            part[start : start + max_chars]
            for start in range(0, len(part), max_chars)
        )
    ]

    chunks = []
    current = ""
    for segment in segments:
        candidate = f"{current}\n{segment}".strip()
        if current and len(candidate) > max_chars:
            chunks.append(current)
            current = segment
        else:
            current = candidate
    if current:
        chunks.append(current)

    valid_chunks = [chunk for chunk in chunks if len(chunk) >= 40]
    return valid_chunks or chunks


def build_retrieval_corpus(
    legal_units_path: str = "data/processed/legal_units.parquet",
    output_path: str = "data/processed/retrieval_corpus.parquet",
    max_chars: int = 1800,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.{os.getpid()}.tmp.parquet")
    writer = pq.ParquetWriter(temporary, RETRIEVAL_SCHEMA, compression="zstd")
    buffered_rows: list[dict] = []
    unit_count = 0
    chunk_count = 0

    try:
        parquet = pq.ParquetFile(legal_units_path)
        for batch in parquet.iter_batches(batch_size=2048):
            for raw in batch.to_pylist():
                parts = split_article_text(raw.get("text"), max_chars=max_chars)
                if not parts:
                    continue
                unit_count += 1
                for part_index, content in enumerate(parts):
                    row = {
                        "chunk_id": (
                            f"{raw['unit_id']}::part::{part_index:03d}"
                        ),
                        "unit_id": raw["unit_id"],
                        "chunk_type": "article_part",
                        "text": build_article_chunk(raw, content=content),
                        "content_text": content,
                        "part_index": part_index,
                        "part_count": len(parts),
                        "is_current": bool(raw.get("is_current")),
                    }
                    for column in RETRIEVAL_STRING_COLUMNS:
                        if column not in row:
                            row[column] = str(raw.get(column) or "")
                    buffered_rows.append(row)
                    chunk_count += 1

                    if len(buffered_rows) >= 10_000:
                        writer.write_table(
                            pa.Table.from_pylist(
                                buffered_rows, schema=RETRIEVAL_SCHEMA
                            )
                        )
                        buffered_rows.clear()

        if buffered_rows:
            writer.write_table(
                pa.Table.from_pylist(buffered_rows, schema=RETRIEVAL_SCHEMA)
            )
    finally:
        writer.close()

    temporary.replace(output)
    print(
        f"Saved {chunk_count} retrieval chunks from {unit_count} legal units "
        f"to {output}"
    )
    return output


if __name__ == "__main__":
    build_retrieval_corpus()
