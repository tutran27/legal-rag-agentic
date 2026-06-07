import pandas as pd
import json
from pathlib import Path
import re


ARTICLE_PREFIX_RE = re.compile(r"^Điều\s+\d+[a-zA-Z]?\s+", flags=re.IGNORECASE)
DATE_SUFFIX_RE = re.compile(r"\s+ngày\s+\d{1,2}/\d{1,2}/\d{4}.*$", flags=re.IGNORECASE)


def normalize_text(value) -> str:
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalize_source_links(value):
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, list):
        return value
    return [value] if value else []


def clean_doc_title_raw(value: str) -> str:
    text = normalize_text(value).strip("() ")
    text = ARTICLE_PREFIX_RE.sub("", text)
    text = DATE_SUFFIX_RE.sub("", text)
    text = re.sub(r",?\s*có hiệu lực thi hành kể từ.*$", "", text, flags=re.IGNORECASE)
    return normalize_text(text)


def build_doc_name_for_submission(doc_type: str, doc_code: str, title: str) -> str:
    parts = [normalize_text(doc_type), normalize_text(doc_code), normalize_text(title)]
    return normalize_text(" ".join([p for p in parts if p]))


def load_canonical_documents(documents_path: str | None) -> dict[str, list[dict]]:
    if not documents_path or not Path(documents_path).exists():
        return {}

    docs = pd.read_parquet(documents_path)
    docs = docs.dropna(subset=["doc_code"])

    canonical = {}
    for _, row in docs.iterrows():
        r = row.to_dict()
        doc_code = normalize_text(r.get("doc_code"))
        if not doc_code:
            continue

        doc_type = normalize_text(r.get("doc_type"))
        doc_title_raw = normalize_text(r.get("doc_title_raw"))
        doc_name = normalize_text(r.get("doc_title_submission"))

        if not doc_name:
            doc_name = build_doc_name_for_submission(doc_type, doc_code, doc_title_raw)

        canonical.setdefault(doc_code, []).append({
            "doc_id": normalize_text(r.get("doc_id")),
            "doc_code": doc_code,
            "doc_type": doc_type,
            "doc_title_raw": doc_title_raw,
            "doc_title_submission": doc_name,
            "doc_name_for_submission": doc_name,
            "issuer": normalize_text(r.get("issuer")),
            "status": normalize_text(r.get("status")),
            "source_url": normalize_text(r.get("source_url")),
        })

    return canonical


def build_fallback_documents(df: pd.DataFrame) -> dict[str, dict]:
    fallback = {}
    for _, row in df.iterrows():
        r = row.to_dict()
        doc_code = normalize_text(r.get("doc_code"))
        if not doc_code:
            continue

        doc_type = normalize_text(r.get("doc_type"))
        raw_title = clean_doc_title_raw(r.get("doc_title_raw") or r.get("source_note_text"))
        doc_name = build_doc_name_for_submission(doc_type, doc_code, raw_title)
        candidate = {
            "doc_id": "",
            "doc_code": doc_code,
            "doc_type": doc_type,
            "doc_title_raw": raw_title,
            "doc_title_submission": doc_name,
            "doc_name_for_submission": doc_name,
            "issuer": "",
            "status": "",
            "source_url": "",
        }

        current = fallback.get(doc_code)
        current_score = fallback_doc_score(current) if current else -1
        candidate_score = fallback_doc_score(candidate)
        if candidate_score > current_score:
            fallback[doc_code] = candidate

    return fallback


def fallback_doc_score(doc: dict) -> int:
    title = normalize_text(doc.get("doc_title_raw"))
    doc_type = normalize_text(doc.get("doc_type"))
    score = len(title)
    if doc_type:
        score += 1000
    if "có hiệu lực thi hành kể từ" in title.lower():
        score -= 500
    return score


def select_canonical_doc(
    candidates: list[dict],
    fallback: dict,
    source_doc_type: str,
    source_doc_title: str,
) -> dict:
    if not candidates:
        return fallback
    if len(candidates) == 1:
        return candidates[0]

    source_doc_type = normalize_text(source_doc_type).lower()
    source_doc_title = clean_doc_title_raw(source_doc_title).lower()

    def score(candidate: dict) -> int:
        candidate_type = normalize_text(candidate.get("doc_type")).lower()
        candidate_title = normalize_text(candidate.get("doc_title_raw")).lower()
        candidate_name = normalize_text(candidate.get("doc_name_for_submission")).lower()

        value = 0
        if source_doc_type and candidate_type == source_doc_type:
            value += 100
        if source_doc_type and source_doc_type in candidate_type:
            value += 30
        if candidate_title and candidate_title in source_doc_title:
            value += 60
        if candidate_title and source_doc_title and source_doc_title in candidate_name:
            value += 40
        return value

    best = max(candidates, key=score)
    return best if score(best) > 0 else fallback

def build_search_text(row):
    parts=[
        row.get("doc_name_for_submission") or row.get("doc_title_submission", ""),
        row.get("article", ""),
        row.get("article_title", ""),
        row.get("text") or row.get("content_text", ""),
        row.get("chapter_title", ""),
        row.get("subject_title", ""),
        row.get("topic_title", ""),
    ]
    return "\n".join([str(p) for p in parts if p])

def build_legal_units_from_phapdien(
    input_path: str= "data/processed/phapdien-moj-gov-vn.parquet",
    output_path: str= "data/processed/legal_units.parquet",
    documents_path: str | None = "data/processed/documents.parquet",
):
    df = pd.read_parquet(input_path)
    canonical_docs = load_canonical_documents(documents_path)
    fallback_docs = build_fallback_documents(df)
    
    rows=[]

    for _, row in df.iterrows():
        r=row.to_dict()

        article_title = r.get("article_title", "")
        doc_code = r.get("doc_code", "")
        article = normalize_text(r.get("article"))
        
        if not article_title or not doc_code or not article:
            continue

        doc_code = normalize_text(doc_code)
        article_title = normalize_text(article_title)
        source_article_id = normalize_text(r.get("article_id"))
        doc = select_canonical_doc(
            canonical_docs.get(doc_code, []),
            fallback_docs.get(doc_code, {}),
            r.get("doc_type"),
            r.get("doc_title_raw") or r.get("source_note_text"),
        )
        doc_name = normalize_text(doc.get("doc_name_for_submission"))
        parent_path = " > ".join(
            [
                p
                for p in [
                    normalize_text(r.get("topic_title")),
                    normalize_text(r.get("subject_title")),
                    normalize_text(r.get("chapter_title")),
                ]
                if p
            ]
        )

        unit_id = f"{doc_code}|{article}|{source_article_id or article_title}|phapdien"

        metadata={
            "article_id": source_article_id,
            "doc_id": normalize_text(doc.get("doc_id")),
            "doc_code": doc_code,
            "doc_type": normalize_text(doc.get("doc_type") or r.get("doc_type")),
            "doc_title_raw": normalize_text(doc.get("doc_title_raw")),
            "doc_title_submission": doc_name,
            "doc_name_for_submission": doc_name,
            "article": article,
            "article_title": article_title,
            "parent_path": parent_path,
            "topic_title": normalize_text(r.get("topic_title")),
            "subject_title": normalize_text(r.get("subject_title")),
            "chapter_title": normalize_text(r.get("chapter_title")),
            "domain": normalize_text(r.get("topic_title")),
            "issuer": normalize_text(doc.get("issuer")),
            "status": normalize_text(doc.get("status")),
            "effective_from": "",
            "effective_to": "",
            "source_dataset": normalize_text(r.get("source_dataset")),
            "source_url": normalize_text(r.get("source_url")) or normalize_text(doc.get("source_url")),
            "source_note_text": normalize_text(r.get("source_note_text")),
            "related_note_text": normalize_text(r.get("related_note_text")),
            "source_links": normalize_source_links(r.get("source_links")),
        }
        
        legal_unit = {
            "unit_id": unit_id,
            "article_id": metadata["article_id"],
            "unit_type": "article",
            "doc_id": metadata["doc_id"],
            "doc_code": doc_code,
            "doc_type": metadata["doc_type"],
            "doc_title_raw": metadata["doc_title_raw"],
            "doc_title_submission": doc_name,
            "doc_name_for_submission": doc_name,
            "article": article,
            "article_title": article_title,
            "parent_path": parent_path,
            "topic_title": metadata["topic_title"],
            "subject_title": metadata["subject_title"],
            "chapter_title": metadata["chapter_title"],
            "domain": metadata["domain"],
            "status": metadata["status"],
            "effective_from": metadata["effective_from"],
            "effective_to": metadata["effective_to"],
            "text": normalize_text(r.get("content_text")),
            "source_dataset": metadata["source_dataset"],
            "source_url": metadata["source_url"],
            "source_note_text": metadata["source_note_text"],
            "metadata_json": json.dumps(metadata, ensure_ascii=False, indent=2),
        }
        legal_unit["search_text"] = build_search_text(legal_unit)

        rows.append({
            **legal_unit,
        })
    
    out = pd.DataFrame(rows)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Saved {len(out)} legal units to {output_path}")
    return out


if __name__ == "__main__":
    build_legal_units_from_phapdien()
