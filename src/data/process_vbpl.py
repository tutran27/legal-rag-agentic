from __future__ import annotations

import html
import os
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


ARTICLE_HEADING_RE = re.compile(
    r"^(?P<article>Điều\s+(?:\d+[a-zA-Z]?|duy nhất))"
    r"(?:(?:\s*[.:\-]\s*|\s+)(?P<title>.*))?$",
    flags=re.IGNORECASE,
)
PARENT_HEADING_RE = re.compile(
    r"^(Phần|Chương|Mục|Tiểu mục)\s+([A-ZÀ-Ỹ0-9IVXLCDM]+)(?:[.:\-]\s*(.*))?$",
    flags=re.IGNORECASE,
)
SCRIPT_STYLE_RE = re.compile(
    r"<(?:script|style)\b[^>]*>.*?</(?:script|style)>",
    flags=re.IGNORECASE | re.DOTALL,
)
BLOCK_TAG_RE = re.compile(
    r"</?(?:p|div|tr|td|th|li|br|h[1-6]|table|section|article)\b[^>]*>",
    flags=re.IGNORECASE,
)
HTML_TAG_RE = re.compile(r"<[^>]+>")

VBPL_ARTICLE_SCHEMA = pa.schema(
    [
        ("article_id", pa.string()),
        ("doc_id", pa.string()),
        ("doc_code", pa.string()),
        ("doc_type", pa.string()),
        ("doc_title_raw", pa.string()),
        ("doc_name_for_submission", pa.string()),
        ("article", pa.string()),
        ("article_title", pa.string()),
        ("parent_path", pa.string()),
        ("content_text", pa.string()),
        ("issuer", pa.string()),
        ("issued_date", pa.string()),
        ("effective_from", pa.string()),
        ("effective_to", pa.string()),
        ("status", pa.string()),
        ("domain", pa.string()),
        ("sector", pa.string()),
        ("source_dataset", pa.string()),
        ("source_url", pa.string()),
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
    return re.sub(r"[ \t\r\f\v]+", " ", str(value)).strip()


def normalize_multiline(value) -> str:
    lines = [normalize_text(line) for line in str(value or "").splitlines()]
    return "\n".join(line for line in lines if line)


def normalize_date(value) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    parsed = pd.to_datetime(text, dayfirst=True, errors="coerce")
    if pd.isna(parsed):
        return text
    return parsed.date().isoformat()


def build_doc_name_for_submission(doc_type: str, doc_code: str, title: str) -> str:
    return normalize_text(" ".join(p for p in [doc_type, doc_code, title] if p))


def normalize_article(value: str) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    return "Điều " + text.split(maxsplit=1)[1]


def build_documents(
    metadata_path: str = "data/raw/vbpl_large/metadata.parquet",
    output_path: str = "data/processed/documents.parquet",
) -> pd.DataFrame:
    metadata = pd.read_parquet(metadata_path)
    rows = []

    for row in metadata.itertuples(index=False):
        raw = row._asdict()
        doc_id = normalize_text(raw.get("id"))
        doc_code = normalize_text(raw.get("so_ky_hieu"))
        doc_type = normalize_text(raw.get("loai_van_ban"))
        title = normalize_text(raw.get("title"))

        rows.append(
            {
                "doc_id": doc_id,
                "doc_code": doc_code,
                "doc_type": doc_type,
                "doc_title_raw": title,
                "doc_title_submission": build_doc_name_for_submission(
                    doc_type, doc_code, title
                ),
                "issuer": normalize_text(raw.get("co_quan_ban_hanh")),
                "issued_date": normalize_date(raw.get("ngay_ban_hanh")),
                "effective_from": normalize_date(raw.get("ngay_co_hieu_luc")),
                "effective_to": normalize_date(raw.get("ngay_het_hieu_luc")),
                "status": normalize_text(raw.get("tinh_trang_hieu_luc")),
                "domain": normalize_text(raw.get("linh_vuc")),
                "sector": normalize_text(raw.get("nganh")),
                "scope": normalize_text(raw.get("pham_vi")),
                "source_dataset": "th1nhng0/vietnamese-legal-documents",
                "source_url": "",
            }
        )

    out = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"Saved {len(out)} documents to {output_path}")
    return out


def html_to_lines(content_html: str) -> list[str]:
    if not normalize_text(content_html):
        return []
    text = SCRIPT_STYLE_RE.sub(" ", str(content_html))
    text = BLOCK_TAG_RE.sub("\n", text)
    text = HTML_TAG_RE.sub(" ", text)
    text = html.unescape(text)
    return [
        line
        for line in (normalize_text(x) for x in text.splitlines())
        if line
    ]


def parse_articles_from_html(content_html: str) -> list[dict[str, str]]:
    lines = html_to_lines(content_html)
    articles: list[dict[str, str]] = []
    parent_parts: list[str] = []
    current: dict[str, object] | None = None
    pending_parent_index: int | None = None

    for line in lines:
        parent_match = PARENT_HEADING_RE.match(line)
        if parent_match:
            heading_type = parent_match.group(1).title()
            heading_value = parent_match.group(2)
            heading_title = normalize_text(parent_match.group(3))
            parent_label = normalize_text(
                f"{heading_type} {heading_value}"
                + (f" - {heading_title}" if heading_title else "")
            )

            level = {"Phần": 0, "Chương": 1, "Mục": 2, "Tiểu Mục": 3}.get(
                heading_type, 3
            )
            parent_parts = parent_parts[:level]
            parent_parts.append(parent_label)
            pending_parent_index = len(parent_parts) - 1
            continue

        if pending_parent_index is not None and line.isupper():
            parent_parts[pending_parent_index] = (
                f"{parent_parts[pending_parent_index]} - {line}"
            )
            pending_parent_index = None
            continue
        pending_parent_index = None

        article_match = ARTICLE_HEADING_RE.match(line)
        if article_match:
            if current is not None:
                articles.append(_finish_article(current))
            current = {
                "article": normalize_article(article_match.group("article")),
                "article_title": normalize_text(article_match.group("title")),
                "parent_path": " > ".join(parent_parts),
                "content_lines": [],
            }
            continue

        if current is not None:
            if (
                not current["article_title"]
                and not current["content_lines"]
                and not re.match(r"^(?:\d+[.)]|[a-zđ]\))\s*", line, re.IGNORECASE)
            ):
                current["article_title"] = line
                continue
            current["content_lines"].append(line)

    if current is not None:
        articles.append(_finish_article(current))

    return [
        article
        for article in articles
        if len(article["content_text"]) >= 20
    ]


def _finish_article(current: dict[str, object]) -> dict[str, str]:
    return {
        "article": normalize_text(current["article"]),
        "article_title": normalize_text(current["article_title"]),
        "parent_path": normalize_text(current["parent_path"]),
        "content_text": normalize_multiline("\n".join(current["content_lines"])),
    }


def build_vbpl_articles(
    metadata_path: str = "data/raw/vbpl_large/metadata.parquet",
    content_path: str = "data/raw/vbpl_large/content.parquet",
    output_path: str = "data/processed/vbpl_articles.parquet",
    batch_size: int = 512,
) -> Path:
    metadata = pd.read_parquet(metadata_path)
    metadata_by_id = {
        normalize_text(row.id): row._asdict()
        for row in metadata.itertuples(index=False)
    }

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.{os.getpid()}.tmp.parquet")
    writer = pq.ParquetWriter(temporary, VBPL_ARTICLE_SCHEMA, compression="zstd")

    parquet = pq.ParquetFile(content_path)
    max_html_length: dict[str, int] = {}
    for batch in parquet.iter_batches(
        batch_size=batch_size, columns=["id", "content_html"]
    ):
        for content_row in batch.to_pylist():
            doc_id = normalize_text(content_row.get("id"))
            html_length = len(str(content_row.get("content_html") or ""))
            if doc_id and html_length > max_html_length.get(doc_id, -1):
                max_html_length[doc_id] = html_length

    seen_doc_ids: set[str] = set()
    buffered_rows: list[dict] = []
    article_count = 0
    document_count = 0

    try:
        for batch in parquet.iter_batches(
            batch_size=batch_size, columns=["id", "content_html"]
        ):
            for content_row in batch.to_pylist():
                doc_id = normalize_text(content_row.get("id"))
                if not doc_id or doc_id in seen_doc_ids:
                    continue
                html = content_row.get("content_html")
                if len(str(html or "")) < max_html_length.get(doc_id, 0):
                    continue
                seen_doc_ids.add(doc_id)

                raw = metadata_by_id.get(doc_id)
                if not raw:
                    continue
                doc_code = normalize_text(raw.get("so_ky_hieu"))
                doc_type = normalize_text(raw.get("loai_van_ban"))
                title = normalize_text(raw.get("title"))
                doc_name = build_doc_name_for_submission(doc_type, doc_code, title)
                if not doc_code or not doc_name:
                    continue

                articles = parse_articles_from_html(html)
                if articles:
                    document_count += 1
                for article in articles:
                    buffered_rows.append(
                        {
                            "article_id": f"vbpl::{doc_id}::{article['article']}",
                            "doc_id": doc_id,
                            "doc_code": doc_code,
                            "doc_type": doc_type,
                            "doc_title_raw": title,
                            "doc_name_for_submission": doc_name,
                            "article": article["article"],
                            "article_title": article["article_title"],
                            "parent_path": article["parent_path"],
                            "content_text": article["content_text"],
                            "issuer": normalize_text(raw.get("co_quan_ban_hanh")),
                            "issued_date": normalize_date(raw.get("ngay_ban_hanh")),
                            "effective_from": normalize_date(
                                raw.get("ngay_co_hieu_luc")
                            ),
                            "effective_to": normalize_date(
                                raw.get("ngay_het_hieu_luc")
                            ),
                            "status": normalize_text(
                                raw.get("tinh_trang_hieu_luc")
                            ),
                            "domain": normalize_text(raw.get("linh_vuc")),
                            "sector": normalize_text(raw.get("nganh")),
                            "source_dataset": (
                                "th1nhng0/vietnamese-legal-documents"
                            ),
                            "source_url": "",
                        }
                    )
                    article_count += 1

                if len(buffered_rows) >= 10_000:
                    writer.write_table(
                        pa.Table.from_pylist(
                            buffered_rows, schema=VBPL_ARTICLE_SCHEMA
                        )
                    )
                    buffered_rows.clear()

        if buffered_rows:
            writer.write_table(
                pa.Table.from_pylist(buffered_rows, schema=VBPL_ARTICLE_SCHEMA)
            )
    finally:
        writer.close()

    temporary.replace(output)
    print(
        f"Saved {article_count} VBPL articles from {document_count} documents "
        f"to {output}"
    )
    return output


def normalize_relation_type(raw) -> str:
    text = normalize_text(raw).lower()
    if "sửa đổi" in text or "amend" in text:
        return "AMENDS"
    if "thay thế" in text or "replace" in text:
        return "REPLACES"
    if "bãi bỏ" in text or "repeal" in text:
        return "REPEALS"
    if "hướng dẫn" in text or "guide" in text:
        return "GUIDES"
    if "trích dẫn" in text or "cite" in text:
        return "CITES"
    if "căn cứ" in text:
        return "REFERENCES"
    return "RELATED"


def build_legal_edges(
    relationships_path: str = "data/raw/vbpl_large/relationships.parquet",
    output_path: str = "data/processed/legal_edges.parquet",
) -> pd.DataFrame:
    relationships = pd.read_parquet(relationships_path)
    rows = []

    for row in relationships.itertuples(index=False):
        raw = row._asdict()
        source_id = normalize_text(
            raw.get("doc_id") or raw.get("source_doc_id") or raw.get("id")
        )
        target_id = normalize_text(
            raw.get("target_doc_id")
            or raw.get("target_id")
            or raw.get("other_doc_id")
        )
        relation_raw = normalize_text(
            raw.get("relation_type") or raw.get("type") or raw.get("relationship")
        )
        if not source_id or not target_id:
            continue
        rows.append(
            {
                "source_doc_id": source_id,
                "target_doc_id": target_id,
                "relation_type": normalize_relation_type(relation_raw),
                "relation_raw": relation_raw,
            }
        )

    out = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)
    print(f"Saved {len(out)} legal edges to {output_path}")
    return out


if __name__ == "__main__":
    build_documents()
    build_vbpl_articles()
    build_legal_edges()
