import pandas as pd
import json
from pathlib import Path

def build_search_text(row):
    parts=[
        row.get("doc_title_submission", ""),        
        row.get("article_title", ""),
        row.get("content_text", ""),
        row.get("chapter_title", ""),
        row.get("subject_title", ""),
        row.get("topic_title", ""),
    ]
    return "\n".join([str(p) for p in parts if p])

def build_legal_units_from_phapdien(
    input_path: str= "data/processed/phapdien-moj-gov-vn.parquet",
    output_path: str= "data/processed/legal_units.parquet",
):
    df = pd.read_parquet(input_path)
    
    rows=[]

    for _, row in df.iterrows():
        r=row.to_dict()

        article_title = r.get("article_title", "")
        doc_code = r.get("doc_code", "")
        
        if not article_title or not doc_code:
            continue

        unit_id = f"{doc_code}|{article_title}|phapdien"

        metadata={
            "topic_title": r.get("topic_title", ""),
            "subject_title": r.get("subject_title", ""),
            "chapter_title": r.get("chapter_title", ""),
            "doc_title_submission": r.get("doc_title_submission", ""),
            "related_note_text": r.get("related_note_text", ""),
            "source_links": (r.get("source_links", "")).tolist(),
        }
        
        rows.append({
            "unit_id": unit_id,
            "unit_type": "article",
            "doc_code": doc_code,
            "doc_type": r.get("doc_type"),
            "doc_title_submission": r.get("doc_title_submission"),
            "article": r.get("article"),
            "article_title": r.get("article_title"),
            "text": r.get("content_text"),
            "search_text": build_search_text(r),
            "source_dataset": r.get("source_dataset"),
            "source_url": r.get("source_url"),
            "metadata_json": json.dumps(metadata, ensure_ascii=False, indent=2),
        })
    
    out = pd.DataFrame(rows)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Saved {len(out)} legal units to {output_path}")
    return out


if __name__ == "__main__":
    build_legal_units_from_phapdien()