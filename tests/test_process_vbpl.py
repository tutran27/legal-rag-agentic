import pandas as pd


def test_documents_have_required_columns():
    df = pd.read_parquet("data/processed/documents.parquet")

    required = [
        "doc_id",
        "doc_code",
        "doc_type",
        "doc_title_submission",
        "status",
        "source_url",
    ]

    for col in required:
        assert col in df.columns


def test_legal_edges_have_required_columns():
    df = pd.read_parquet("data/processed/legal_edges.parquet")

    required = [
        "source_doc_id",
        "target_doc_id",
        "relation_type",
    ]

    for col in required:
        assert col in df.columns