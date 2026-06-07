import json

#  Dùng cho rerank 
def build_article_parent_chunk(row):
    metadata = json.loads(row.get("metadata_json") or "{}")

    return f"""
[VĂN BẢN]
{row.get("doc_name_for_submission") or row["doc_title_submission"]}

[LĨNH VỰC / CHỦ ĐỀ]
{metadata.get("domain", "") or metadata.get("topic_title", "")} > {metadata.get("subject_title", "")}

[CHƯƠNG / MỤC]
{metadata.get("parent_path", "") or metadata.get("chapter_title", "")}

[ĐIỀU]
{row["article"]}

[NỘI DUNG]
{row["text"]}

[GHI CHÚ NGUỒN]
{metadata.get("source_note_text", "")}
""".strip()

if __name__ == "__main__":
    import pandas as pd
    df = pd.read_parquet("data/processed/legal_units.parquet")
    print(build_article_parent_chunk(df.iloc[5]))

"""
[VĂN BẢN]
Luật 32/2004/QH11 Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia

[LĨNH VỰC / CHỦ ĐỀ]
An ninh Quốc gia > An ninh Quốc gia

[CHƯƠNG / MỤC]
Chương I: Nguyên tắc chung

[ĐIỀU]
Điều 1

[NỘI DUNG]
Luật này quy định về chính sách an ninh quốc gia; nguyên tắc, nhiệm vụ, biện pháp bảo vệ an ninh quốc gia; quyền, nghĩa vụ, trách nhiệm của cơ quan, tổ chức, công dân trong bảo vệ an ninh quốc gia.

[GHI CHÚ NGUỒN]
"""
