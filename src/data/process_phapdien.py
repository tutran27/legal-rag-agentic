import pandas as pd
from pathlib import Path
import re

DOC_CODE_PATTERN = re.compile(
    r"\b(?P<code>\d{1,3}/\d{4}/[A-ZĐ0-9]+(?:-[A-ZĐ0-9]+)*)\b",
    flags=re.IGNORECASE,
)

ARTICLE_PATTERN = re.compile(
    r"\b(Điều\s+\d+[a-zA-Z]?)\b",
    flags=re.IGNORECASE,
)

def normalize_space(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()

def extract_original_article(source_note_text: str) -> str:
    # Extract article from source_note_text
    # Example: "(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia ngày 03/12/2004 của Quốc hội, có hiệu lực thi hành kể từ ngày 01/07/2005 )"
    # Should return "Điều 1"
    match = ARTICLE_PATTERN.search(source_note_text)
    if match:
        return match.group(1).replace("điều", "Điều").strip()
    return ""

def extract_doc_code(source_note_text: str) -> str:
    # Extract doc code from source_note_text
    # Example: "(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia ngày 03/12/2004 của Quốc hội, có hiệu lực thi hành kể từ ngày 01/07/2005 )"
    # Should return "32/2004/QH11"
    match = DOC_CODE_PATTERN.search(source_note_text)
    if match:
        return match.group("code").strip().replace("luật số", "Luật số")
    return ""

def infer_doc_type(source_note_text: str) -> str:
    # Infer doc type from source_note_text
    # Example: "(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia ngày 03/12/2004 của Quốc hội, có hiệu lực thi hành kể từ ngày 01/07/2005 )"
    # Should return "Luật"
    match = re.search(r"(Luật|Nghị định|Thông tư|Quyết định|Nghị quyết|Bộ luật|Bộ luật hình sự|Bộ luật dân sự|Bộ luật lao động|Bộ luật thương mại|Bộ luật kinh tế|Bộ luật hình sự|Bộ luật dân sự|Bộ luật lao động|Bộ luật thương mại|Bộ luật kinh tế)", source_note_text)
    if match:
        return match.group(1).strip()
    return ""

def extract_doc_title_rough(source_note_text: str | None) -> str | None:
    """
    Bản rough. Sau này nên map lại với VBPL metadata để có title chuẩn.
    """
    text = normalize_space(source_note_text)
    if not text:
        return None

    # Bỏ ngoặc ngoài
    text = text.strip("() ")

    # Cắt sau ngày nếu có
    text = re.split(r"\s+ngày\s+\d{1,2}/\d{1,2}/\d{4}", text, flags=re.IGNORECASE)[0]

    return text

def build_submission_title(doc_type: str | None, doc_code: str | None, raw_title: str | None) -> str | None:
    if not doc_type or not doc_code:
        return raw_title

    raw_title = normalize_space(raw_title)

    # Nếu raw_title đã quá dài, vẫn giữ để tránh mất trích yếu.
    # Sau này có thể canonicalize bằng VBPL metadata.
    return f"{doc_type} {doc_code} {raw_title}".strip()

def process_phapdien_article(
    input_path: str = "data/raw/phapdien/articles_train.parquet",
    output_path: str = "data/processed/phapdien-moj-gov-vn.parquet"
):
    df=pd.read_parquet(input_path)
    
    rows=[]
    
    for idx, row in df.iterrows():
        source_note = row.get("source_note_text", "")
        content_text = row.get("content_text", "")
        article_title = row.get("article_title", "")

        if not source_note or not content_text:
            continue
        article_id = f"phapdien::{idx}"
        # Lấy Điều gốc
        origin_article = extract_original_article(source_note)
        
        # Lấy mã văn bản và loại văn bản
        doc_code = extract_doc_code(source_note)
        doc_type = infer_doc_type(source_note)

        # Lấy tiêu đề văn bản
        raw_title = extract_doc_title_rough(source_note)
        doc_title_submission = build_submission_title(doc_type, doc_code, raw_title)
        
        rows.append({
            "article_id": article_id,
            "source_dataset": "tmquan/phapdien-moj-gov-vn",

            "topic_title": row["topic_title"],
            "subject_title": row["subject_title"],
            "chapter_title": row["chapter_title"],
            
            "phapdien_article_title": article_title,
            "article_title": article_title,
            "article": origin_article,

            "doc_code": doc_code,
            "doc_type": doc_type,
            "doc_title_raw": raw_title,
            "doc_title_submission": doc_title_submission,

            "origin_article": origin_article,
            "article_title": article_title,

            "source_note_text": source_note,
            "source_links": row.get("source_links"),
            "related_note_text": row.get("related_note_text"),
            "content_text": content_text,
            "source_url": row.get("source_url"),
        })

    df_out = pd.DataFrame(rows)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)

    print(f"Saved {len(df_out)} phapdien articles to {output_path}")
    return df_out

if __name__ == "__main__":
    source_note_text = "(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia ngày 03/12/2004 của Quốc hội, có hiệu lực thi hành kể từ ngày 01/07/2005 )"
    
    print("------------ Test extract doc to code ------------")
    print(f"Source note text: {source_note_text}")
    print(f"Origin article: {extract_original_article(source_note_text)}")
    print(f"Doc code: {extract_doc_code(source_note_text)}")
    print(f"Doc type: {infer_doc_type(source_note_text)}")
    print(f"Doc title rough: {extract_doc_title_rough(source_note_text)}")
    print(f"Doc title submission: {build_submission_title(infer_doc_type(source_note_text), extract_doc_code(source_note_text), extract_doc_title_rough(source_note_text))}")
    
    # Test process phapdien
    print("------------ Test process phapdien ------------")
    df = process_phapdien_article()
    print(df.head())
    