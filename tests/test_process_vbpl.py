import pandas as pd

from src.data.process_vbpl import parse_articles_from_html


def test_parse_articles_from_html():
    html = """
    <p>Chương I</p>
    <p>QUY ĐỊNH CHUNG</p>
    <p><strong>Điều 1. Phạm vi điều chỉnh</strong></p>
    <p>Luật này quy định nội dung thử nghiệm đủ dài để tạo một điều luật.</p>
    <p><strong>Điều 2. Đối tượng áp dụng</strong></p>
    <p>Áp dụng cho các tổ chức và cá nhân có liên quan đến nội dung thử nghiệm.</p>
    """

    articles = parse_articles_from_html(html)

    assert len(articles) == 2
    assert articles[0]["article"] == "Điều 1"
    assert articles[0]["article_title"] == "Phạm vi điều chỉnh"
    assert "nội dung thử nghiệm" in articles[0]["content_text"]
    assert articles[0]["parent_path"].startswith("Chương I")


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
