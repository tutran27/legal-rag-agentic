# Dùng cho relevant_articles (tìm kiếm theo điều luật)

def build_article_chunk(row):
    return f"""
[VĂN BẢN]
{row.get("doc_name_for_submission") or row["doc_title_submission"]}

[ĐIỀU]
{row["article"]}

[TIÊU ĐỀ ĐIỀU]
{row.get("article_title") or ""}

[NỘI DUNG]
{row["text"]}
""".strip()

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_parquet("data/processed/legal_units.parquet")
    print(build_article_chunk(df.iloc[0]))

""" SAMPLE

[VĂN BẢN]
Luật 32/2004/QH11 Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia

[ĐIỀU]
Điều 1

[TIÊU ĐỀ ĐIỀU]
Điều 1.1.LQ.1. Phạm vi điều chỉnh

[NỘI DUNG]
Luật này quy định về chính sách an ninh quốc gia; nguyên tắc, nhiệm vụ, biện pháp bảo vệ an ninh quốc gia; quyền, nghĩa vụ, trách nhiệm của cơ quan, tổ chức, công dân trong bảo vệ an ninh quốc gia.
"""
