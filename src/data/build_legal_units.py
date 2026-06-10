from __future__ import annotations

import os
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ARTICLE_PREFIX_RE = re.compile(r"^Điều\s+\d+[a-zA-Z]?\s+", flags=re.IGNORECASE)
DATE_SUFFIX_RE = re.compile(
    r"\s+ngày\s+\d{1,2}/\d{1,2}/\d{4}.*$", flags=re.IGNORECASE
)

MAX_ARTICLE_CHARS = 100_000
APPENDIX_HEADING_RE = re.compile(
    r"(?im)^(?:ph\u1ee5\s+l\u1ee5c|danh\s+m\u1ee5c|m\u1eabu\s+s\u1ed1)"
    r"(?:\s|:|$)"
)

LEGAL_UNIT_COLUMNS = [
    "unit_id",
    "article_id",
    "unit_type",
    "doc_id",
    "doc_code",
    "doc_type",
    "doc_title_raw",
    "doc_title_submission",
    "doc_name_for_submission",
    "article",
    "article_title",
    "parent_path",
    "topic_title",
    "subject_title",
    "chapter_title",
    "domain",
    "sector",
    "issuer",
    "issued_date",
    "status",
    "is_current",
    "effective_from",
    "effective_to",
    "text",
    "content_source",
    "source_dataset",
    "source_url",
    "source_note_text",
    "related_note_text",
]
LEGAL_UNIT_SCHEMA = pa.schema(
    [
        (column, pa.bool_() if column == "is_current" else pa.string())
        for column in LEGAL_UNIT_COLUMNS
    ]
)


def normalize_text(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_multiline(value) -> str:
    lines = [normalize_text(line) for line in str(value or "").splitlines()]
    return "\n".join(line for line in lines if line)


def normalize_article_content(value) -> str:
    text = normalize_multiline(value)
    if len(text) <= MAX_ARTICLE_CHARS:
        return text

    appendix = APPENDIX_HEADING_RE.search(text, 200)
    if appendix:
        text = text[: appendix.start()].rstrip()
    return text[:MAX_ARTICLE_CHARS].rstrip()


def normalize_article_key(value) -> str:
    return normalize_text(value).lower().replace(" ", "")


def is_current_status(status: str) -> bool:
    value = normalize_text(status).lower()
    return value in {"", "còn hiệu lực", "hết hiệu lực một phần", "chưa xác định"}


def clean_doc_title_raw(value: str) -> str:
    text = normalize_text(value).strip("() ")
    text = ARTICLE_PREFIX_RE.sub("", text)
    text = DATE_SUFFIX_RE.sub("", text)
    text = re.sub(
        r",?\s*có hiệu lực thi hành kể từ.*$", "", text, flags=re.IGNORECASE
    )
    return normalize_text(text)


def build_doc_name_for_submission(doc_type: str, doc_code: str, title: str) -> str:
    return normalize_text(" ".join(p for p in [doc_type, doc_code, title] if p))


def extract_doc_id(source_links) -> str:
    if hasattr(source_links, "tolist"):
        source_links = source_links.tolist()
    if isinstance(source_links, dict):
        source_links = [source_links]
    for link in source_links or []:
        match = re.search(r"ItemID=(\d+)", str(link.get("href", "")))
        if match:
            return match.group(1)
    return ""


def build_fallback_doc(raw: dict) -> dict:
    doc_code = normalize_text(raw.get("doc_code"))
    doc_type = normalize_text(raw.get("doc_type"))
    title = clean_doc_title_raw(
        raw.get("doc_title_raw") or raw.get("source_note_text")
    )
    doc_name = build_doc_name_for_submission(doc_type, doc_code, title)
    return {
        "doc_id": "",
        "doc_code": doc_code,
        "doc_type": doc_type,
        "doc_title_raw": title,
        "doc_title_submission": doc_name,
        "issuer": "",
        "issued_date": "",
        "effective_from": "",
        "effective_to": "",
        "status": "",
        "domain": "",
        "sector": "",
        "source_url": "",
    }


def prepare_phapdien_units(
    phapdien: pd.DataFrame,
    documents_by_id: dict[str, dict],
    documents_by_code: dict[str, list[dict]],
) -> list[dict]:
    rows = []
    for row in phapdien.itertuples(index=False):
        raw = row._asdict()
        doc_code = normalize_text(raw.get("doc_code"))
        article = normalize_text(raw.get("article"))
        article_title = normalize_text(raw.get("article_title"))
        text = normalize_article_content(raw.get("content_text"))
        if not doc_code or not article or len(text) < 20:
            continue

        doc_id = extract_doc_id(raw.get("source_links"))
        doc = documents_by_id.get(doc_id)
        if not doc:
            candidates = documents_by_code.get(doc_code, [])
            source_type = normalize_text(raw.get("doc_type"))
            doc = next(
                (
                    candidate
                    for candidate in candidates
                    if normalize_text(candidate.get("doc_type")) == source_type
                ),
                candidates[0] if candidates else None,
            )
        if not doc:
            doc = build_fallback_doc(raw)

        topic = normalize_text(raw.get("topic_title"))
        subject = normalize_text(raw.get("subject_title"))
        chapter = normalize_text(raw.get("chapter_title"))
        parent_path = " > ".join(x for x in [topic, subject, chapter] if x)
        source_article_id = normalize_text(raw.get("article_id"))

        rows.append(
            {
                "unit_id": f"phapdien::{source_article_id}",
                "article_id": source_article_id,
                "unit_type": "article",
                "doc_id": normalize_text(doc.get("doc_id")),
                "doc_code": doc_code,
                "doc_type": normalize_text(doc.get("doc_type") or raw.get("doc_type")),
                "doc_title_raw": normalize_text(doc.get("doc_title_raw")),
                "doc_title_submission": normalize_text(
                    doc.get("doc_title_submission")
                ),
                "doc_name_for_submission": normalize_text(
                    doc.get("doc_title_submission")
                ),
                "article": article,
                "article_title": article_title,
                "parent_path": parent_path,
                "topic_title": topic,
                "subject_title": subject,
                "chapter_title": chapter,
                "domain": topic or normalize_text(doc.get("domain")),
                "sector": normalize_text(doc.get("sector")),
                "issuer": normalize_text(doc.get("issuer")),
                "issued_date": normalize_text(doc.get("issued_date")),
                "status": normalize_text(doc.get("status")),
                "is_current": is_current_status(doc.get("status")),
                "effective_from": normalize_text(doc.get("effective_from")),
                "effective_to": normalize_text(doc.get("effective_to")),
                "text": text,
                "content_source": "phapdien",
                "source_dataset": normalize_text(raw.get("source_dataset")),
                "source_url": normalize_text(raw.get("source_url")),
                "source_note_text": normalize_text(raw.get("source_note_text")),
                "related_note_text": normalize_text(raw.get("related_note_text")),
            }
        )
    return rows


def merge_vbpl_article(raw: dict, context: dict) -> dict:
    doc_id = normalize_text(raw.get("doc_id"))
    article = normalize_text(raw.get("article"))
    text = normalize_article_content(raw.get("content_text"))
    doc_name = normalize_text(raw.get("doc_name_for_submission"))
    return {
        "unit_id": f"vbpl::{doc_id}::{normalize_article_key(article)}",
        "article_id": normalize_text(raw.get("article_id")),
        "unit_type": "article",
        "doc_id": doc_id,
        "doc_code": normalize_text(raw.get("doc_code")),
        "doc_type": normalize_text(raw.get("doc_type")),
        "doc_title_raw": normalize_text(raw.get("doc_title_raw")),
        "doc_title_submission": doc_name,
        "doc_name_for_submission": doc_name,
        "article": article,
        "article_title": normalize_text(raw.get("article_title"))
        or normalize_text(context.get("article_title")),
        "parent_path": normalize_text(context.get("parent_path"))
        or normalize_text(raw.get("parent_path")),
        "topic_title": normalize_text(context.get("topic_title")),
        "subject_title": normalize_text(context.get("subject_title")),
        "chapter_title": normalize_text(context.get("chapter_title"))
        or normalize_text(raw.get("parent_path")),
        "domain": normalize_text(context.get("domain"))
        or normalize_text(raw.get("domain")),
        "sector": normalize_text(raw.get("sector")),
        "issuer": normalize_text(raw.get("issuer")),
        "issued_date": normalize_text(raw.get("issued_date")),
        "status": normalize_text(raw.get("status")),
        "is_current": is_current_status(raw.get("status")),
        "effective_from": normalize_text(raw.get("effective_from")),
        "effective_to": normalize_text(raw.get("effective_to")),
        "text": text,
        "content_source": "vbpl",
        "source_dataset": normalize_text(raw.get("source_dataset")),
        "source_url": normalize_text(context.get("source_url"))
        or normalize_text(raw.get("source_url")),
        "source_note_text": normalize_text(context.get("source_note_text")),
        "related_note_text": normalize_text(context.get("related_note_text")),
    }


def build_legal_units(
    phapdien_path: str = "data/processed/phapdien-moj-gov-vn.parquet",
    vbpl_articles_path: str = "data/processed/vbpl_articles.parquet",
    documents_path: str = "data/processed/documents.parquet",
    output_path: str = "data/processed/legal_units.parquet",
    batch_size: int = 2048,
) -> Path:
    phapdien = pd.read_parquet(phapdien_path)
    documents = pd.read_parquet(documents_path).to_dict("records")
    documents_by_id = {normalize_text(doc["doc_id"]): doc for doc in documents}
    documents_by_code: dict[str, list[dict]] = defaultdict(list)
    for doc in documents:
        documents_by_code[normalize_text(doc["doc_code"])].append(doc)

    phapdien_units = prepare_phapdien_units(
        phapdien,
        documents_by_id,
        documents_by_code,
    )
    phapdien_by_key: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for unit in phapdien_units:
        if unit["doc_id"]:
            key = (unit["doc_id"], normalize_article_key(unit["article"]))
            phapdien_by_key[key].append(unit)

    matched_phapdien_ids: set[str] = set()
    seen_unit_ids: set[str] = set()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.{os.getpid()}.tmp.parquet")
    writer = pq.ParquetWriter(temporary, LEGAL_UNIT_SCHEMA, compression="zstd")
    buffered_rows: list[dict] = []
    vbpl_count = 0
    phapdien_count = 0

    try:
        parquet = pq.ParquetFile(vbpl_articles_path)
        for batch in parquet.iter_batches(batch_size=batch_size):
            for raw in batch.to_pylist():
                doc_id = normalize_text(raw.get("doc_id"))
                article = normalize_text(raw.get("article"))
                text = normalize_article_content(raw.get("content_text"))
                if not doc_id or not article or not text:
                    continue

                matches = phapdien_by_key.get(
                    (doc_id, normalize_article_key(article)), []
                )
                context = matches[0] if matches else {}
                unit = merge_vbpl_article(raw, context)
                if unit["unit_id"] in seen_unit_ids:
                    continue
                seen_unit_ids.add(unit["unit_id"])
                for match in matches:
                    matched_phapdien_ids.add(match["article_id"])

                buffered_rows.append(
                    {
                        column: unit.get(column, "")
                        for column in LEGAL_UNIT_COLUMNS
                    }
                )
                vbpl_count += 1
                if len(buffered_rows) >= 10_000:
                    writer.write_table(
                        pa.Table.from_pylist(
                            buffered_rows, schema=LEGAL_UNIT_SCHEMA
                        )
                    )
                    buffered_rows.clear()

        for unit in phapdien_units:
            if unit["article_id"] in matched_phapdien_ids:
                continue
            if unit["unit_id"] in seen_unit_ids:
                continue
            seen_unit_ids.add(unit["unit_id"])
            buffered_rows.append(
                {
                    column: unit.get(column, "")
                    for column in LEGAL_UNIT_COLUMNS
                }
            )
            phapdien_count += 1
            if len(buffered_rows) >= 10_000:
                writer.write_table(
                    pa.Table.from_pylist(buffered_rows, schema=LEGAL_UNIT_SCHEMA)
                )
                buffered_rows.clear()

        if buffered_rows:
            writer.write_table(
                pa.Table.from_pylist(buffered_rows, schema=LEGAL_UNIT_SCHEMA)
            )
    finally:
        writer.close()

    temporary.replace(output)
    print(
        f"Saved {vbpl_count + phapdien_count} legal units to {output} "
        f"(VBPL={vbpl_count}, Phapdien fallback={phapdien_count})"
    )
    return output
if __name__ == "__main__":
    build_legal_units()
