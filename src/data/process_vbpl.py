from pathlib import Path
import pandas as pd

def load_vbpl_raw(
    metadata_path: Path = "data/raw/vbpl_large/metadata_data.parquet",
    content_path: Path = "data/raw/vbpl_large/content_data.parquet",
    relationships_path: Path = "data/raw/vbpl_large/relationships.parquet"
):
    metadata_df = pd.read_parquet(metadata_path)
    content_df = pd.read_parquet(content_path)
    relationships_df = pd.read_parquet(relationships_path)

    return metadata_df, content_df, relationships_df

def inspect_columns():
    meta, cont, rel = load_vbpl_raw()

    print("Metadata columns:", meta.columns.tolist())
    print("Content columns:", cont.columns.tolist())
    print("Relationships columns:", rel.columns.tolist())

def get_first_existing(row, candidates):
    for col in candidates:
        if col in row and pd.notna(row[col]):
            return row[col]
    return None

import re
import pandas as pd
from pathlib import Path


def normalize_text(x):
    if x is None or pd.isna(x):
        return None
    return re.sub(r"\s+", " ", str(x)).strip()


def get_first_existing(row, candidates):
    for col in candidates:
        if col in row and pd.notna(row[col]):
            return row[col]
    return None


def build_documents(
    metadata_path="data/raw/vbpl_large/metadata_data.parquet",
    output_path="data/processed/documents.parquet",
):
    meta = pd.read_parquet(metadata_path)

    rows = []

    for _, row in meta.iterrows():
        row = row.to_dict()

        doc_id = get_first_existing(row, ["id", "doc_id", "document_id"])
        doc_code = get_first_existing(row, ["doc_code", "number", "so_hieu", "code"])
        doc_type = get_first_existing(row, ["doc_type", "type", "loai_van_ban"])
        title = get_first_existing(row, ["title", "trich_yeu", "doc_title", "name"])
        issuer = get_first_existing(row, ["issuer", "co_quan_ban_hanh"])
        status = get_first_existing(row, ["status", "tinh_trang_hieu_luc"])
        source_url = get_first_existing(row, ["source_url", "url"])

        doc_code = normalize_text(doc_code)
        doc_type = normalize_text(doc_type)
        title = normalize_text(title)

        if doc_type and doc_code and title:
            title_submission = f"{doc_type} {doc_code} {title}"
        else:
            title_submission = title

        rows.append({
            "doc_id": str(doc_id) if doc_id is not None else None,
            "doc_code": doc_code,
            "doc_type": doc_type,
            "doc_title_raw": title,
            "doc_title_submission": title_submission,
            "issuer": normalize_text(issuer),
            "status": normalize_text(status),
            "source_url": source_url,
        })

    out = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Saved {len(out)} documents to {output_path}")
    return out

def normalize_relation_type(raw):
    if raw is None:
        return "RELATED"

    s = str(raw).lower()

    if "sửa đổi" in s or "amend" in s:
        return "AMENDS"
    if "thay thế" in s or "replace" in s:
        return "REPLACES"
    if "bãi bỏ" in s or "repeal" in s:
        return "REPEALS"
    if "hướng dẫn" in s or "guide" in s:
        return "GUIDES"
    if "trích dẫn" in s or "cite" in s:
        return "CITES"

    return "RELATED"
    
def build_legal_edges(
    relationships_path="data/raw/vbpl_large/relationships_data.parquet",
    output_path="data/processed/legal_edges.parquet",
):
    rels = pd.read_parquet(relationships_path)

    rows = []

    for _, row in rels.iterrows():
        r = row.to_dict()

        source_id = get_first_existing(r, ["doc_id", "source_doc_id", "source_id"])
        target_id = get_first_existing(r, ["target_doc_id", "target_id", "related_doc_id"])
        relation_raw = get_first_existing(r, ["relation_type", "type", "relationship"])

        if source_id is None or target_id is None:
            continue

        rows.append({
            "source_doc_id": str(source_id),
            "target_doc_id": str(target_id),
            "relation_type": normalize_relation_type(relation_raw),
            "relation_raw": relation_raw,
        })

    out = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Saved {len(out)} legal edges to {output_path}")
    return out
    
if __name__ == "__main__":
    inspect_columns()