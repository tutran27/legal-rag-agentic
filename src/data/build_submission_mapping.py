import pandas as pd
from pathlib import Path


def build_submission_mapping(
    legal_units_path="data/processed/legal_units.parquet",
    output_path="data/processed/submission_mapping.parquet",
):
    df = pd.read_parquet(legal_units_path)

    rows = []

    for _, row in df.iterrows():
        doc_code = row.get("doc_code")
        title = row.get("doc_name_for_submission") or row.get("doc_title_submission")
        article = row.get("article")

        if not doc_code or not title or not article:
            continue

        relevant_doc = f"{doc_code}|{title}"
        relevant_article = f"{doc_code}|{title}|{article}"

        rows.append({
            "unit_id": row.get("unit_id"),
            "relevant_doc": relevant_doc,
            "relevant_article": relevant_article,
        })

    out = pd.DataFrame(rows).drop_duplicates()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Saved {len(out)} mappings to {output_path}")
    return out

if __name__ == "__main__":
    build_submission_mapping()
