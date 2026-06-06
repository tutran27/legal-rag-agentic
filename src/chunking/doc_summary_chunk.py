import pandas as pd
from pathlib import Path

from src.chunking.article_chunk import build_article_chunk
from src.chunking.article_parent_chunk import build_article_parent_chunk

def build_retrieval_corpus(
    legal_units_path="data/processed/legal_units.parquet",
    output_path="data/processed/retrieval_corpus.parquet",
):
    df = pd.read_parquet(legal_units_path)

    rows = []

    for _, row in df.iterrows():
        r = row.to_dict()

        rows.append({
            "chunk_id": r["unit_id"] + "::article",
            "unit_id": r["unit_id"],
            "chunk_type": "article",
            "text": build_article_chunk(r),
        })

        rows.append({
            "chunk_id": r["unit_id"] + "::parent",
            "unit_id": r["unit_id"],
            "chunk_type": "article_parent",
            "text": build_article_parent_chunk(r),
        })

    out = pd.DataFrame(rows)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    print(f"Saved {len(out)} retrieval chunks to {output_path}")
    return out

if __name__ == "__main__":
    build_retrieval_corpus()

"""CHUNK_ID: 32/2004/QH11|Điều 1.1.LQ.1. Phạm vi điều chỉnh|phapdien::article
UNIT_ID: 32/2004/QH11|Điều 1.1.LQ.1. Phạm vi điều chỉnh|phapdien
CHUNK_TYPE: article
TEXT: [VĂN BẢN]
Luật 32/2004/QH11 Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia

[ĐIỀU]
Điều 1

[TIÊU ĐỀ ĐIỀU]
Điều 1.1.LQ.1. Phạm vi điều chỉnh

[NỘI DUNG]
Luật này quy định về chính sách an ninh quốc gia; nguyên tắc, nhiệm vụ, biện pháp bảo vệ an ninh quốc gia; quyền, nghĩa vụ, trách nhiệm của cơ quan, tổ chức, công dân trong bảo vệ an ninh quốc gia."""


"""
CHUNK_ID: 32/2004/QH11|Điều 1.1.LQ.1. Phạm vi điều chỉnh|phapdien::parent
UNIT_ID: 32/2004/QH11|Điều 1.1.LQ.1. Phạm vi điều chỉnh|phapdien
CHUNK_TYPE: article_parent
TEXT: [VĂN BẢN]
Luật 32/2004/QH11 Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia

[LĨNH VỰC / CHỦ ĐỀ]
An ninh quốc gia > An ninh quốc gia

[CHƯƠNG / MỤC]
Chương I - NHỮNG QUY ĐỊNH CHUNG

[ĐIỀU]
Điều 1

[NỘI DUNG]
Luật này quy định về chính sách an ninh quốc gia; nguyên tắc, nhiệm vụ, biện pháp bảo vệ an ninh quốc gia; quyền, nghĩa vụ, trách nhiệm của cơ quan, tổ chức, công dân trong bảo vệ an ninh quốc gia.

[GHI CHÚ NGUỒN]"""