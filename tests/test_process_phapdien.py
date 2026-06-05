from src.data.process_phapdien import (
    extract_original_article,
    extract_doc_code,
    infer_doc_type,
)


def test_extract_original_article():
    text = "(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia ngày 03/12/2004)"
    assert extract_original_article(text) == "Điều 1"


def test_extract_doc_code():
    text = "(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia ngày 03/12/2004)"
    assert extract_doc_code(text) == "32/2004/QH11"


def test_infer_doc_type_law():
    text = "(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia)"
    assert infer_doc_type(text) == "Luật"


def test_infer_doc_type_decree():
    text = "(Điều 1 Nghị định số 80/2021/NĐ-CP Quy định chi tiết...)"
    assert infer_doc_type(text) == "Nghị định"

if __name__ == "__main__":
    test_extract_original_article()
    test_extract_doc_code()
    test_infer_doc_type_law()
    test_infer_doc_type_decree()
    print("All tests passed!")
    
    import pandas as pd

    df = pd.read_parquet("data/processed/phapdien-moj-gov-vn.parquet")

    print("rows:", len(df))
    print("has doc_code:", df["doc_code"].notna().mean())
    print("has article:", df["article"].notna().mean())
    print("has title:", df["doc_title_submission"].notna().mean())