def build_article_chunk(row, content: str | None = None) -> str:
    article_text = content if content is not None else row["text"]
    parent_path = row.get("parent_path") or ""
    status = row.get("status") or ""

    parts = [
        f"[VĂN BẢN]\n{row.get('doc_name_for_submission') or row['doc_title_submission']}",
        f"[ĐIỀU]\n{row['article']}",
        f"[TIÊU ĐỀ ĐIỀU]\n{row.get('article_title') or ''}",
    ]
    if parent_path:
        parts.append(f"[NGỮ CẢNH]\n{parent_path}")
    if status:
        parts.append(f"[HIỆU LỰC]\n{status}")
    parts.append(f"[NỘI DUNG]\n{article_text}")
    return "\n\n".join(parts).strip()
