# Pháp điển articles

Path:
data/raw/vbpl_markdown/default.parquet

Rows:
158,822

Important columns:
- topic_title: chủ đề
- subject_title: đề mục
- article_title: tiêu đề điều pháp điển
- source_note_text: chứa Điều gốc + mã văn bản
- content_text: nội dung điều
- source_url: URL nguồn (Ví dụ thực tế: `https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-24-ld-nd-to-chuc-cac-...`)

Decision:
- Dùng content_text làm text chính
- Dùng source_note_text để parse doc_code và Điều gốc
- Không dùng article_title trực tiếp làm Điều output

Full Sample Details:
- **doc_name**: `1`
- **item_id**: `1`
- **scope**: `trung_uong`
- **source**: `vbpl.vn`
- **source_url**:
  > https://vbpl.vn/van-ban/chi-tiet/nghi-dinh-so-24-ld-nd-to-chuc-cac-co-quan-lao-dong-dia-phuong-lien-khu-va-tinh--1
- **api_url**: `https://vbpl-bientap-gateway.moj.gov.vn/api/qtdc/public/doc/1`
- **title**: `Tổ chức các cơ quan Lao động địa phương liên khu và tỉnh`
- **doc_type**: `nghi_dinh`
- **legal_type**: `Nghị định`
- **legal_area**: `Chưa phân loại`
- **doc_number**: `['24/LĐ-NĐ']`
- **issue_date**: `1950-03-27`
- **year**: `1950.0`
- **issuing_authority**: `Bộ Lao động`
- **summary**: `nan`
- **markdown**:
  > BỘ LAO ĐỘNG CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập – Tự do – Hạnh phúc Số: 24/LĐ-NĐ Hà Nội, ngày 27 tháng 03 năm 1950 NGHỊ ĐỊNH BỘ TRƯỞNG BỘ LAO ĐỘNG Tổ chức các cơ quan Lao động địa phương liên khu và tỉnh BỘ TRƯỞNG BỘ LAO ĐỘNG Chiểu Sắc lệnh số 226/SL ngày 28/11/1946 tổ chức Bộ Lao động trong Chính phủ hiện thời; Chiểu Sắc lệnh số 169/SL ngày 14/04/1948 tổ chức các cơ quan Lao động địa phương; Chiểu Nghị định số 20 ngày 15/05/1948 của Bộ trưởng Bộ Lao động ấn định nhiệm vụ của các Sở và Ty Lao động; NGHỊ ĐỊNH: Điều 1. -Kể từ ngày ký nghị định này, các Sở Lao động sẽ gọi là "Khu Lao động", do một ông Giám đốc Khu Lao động điều khiển. Điều 2. -Ở tỉnh nào xét ra quan hệ nhiều về phương diện lao động sẽ đặt một Ty Lao động, do một Trưởng ty Lao động điều khiển. Ở những tỉnh nào kế cận với nhau mà vấn đề lao động ít quan hệ hơn thì đặt một Ty Lao động liên tỉnh. Ở các tỉnh khác công việc lao động sẽ do Ủy ban Kháng chiến hành chính tỉnh ấy trực tiếp đảm nhận. Điều 3. -Ông Đổng lý Văn phòng Bộ Lao động, các ông Chủ tịch UBKCHC các liên khu và các ông Giám đốc Khu Lao động chiểu nghị định thi hành./. BỘ TRƯỞNG (Đã ký) Nguyễn Văn Tạo
- **num_pages**: `nan`
- **num_sections**: `1`
- **num_paragraphs**: `1`
- **num_sentences**: `6`
- **char_len**: `1149`
- **text_hash**: `6bba1a9f1ebb47741d63ffa6b2d29313`
- **parser_model**: `local/markdownify`
- **parser_runtime**: `local`
- **body_source**: `body_html`
- **parsed_at**: `2026-05-21T10:37:41+00:00`
- **confidence**: `nan`
- **structure_json**:
  > {"schema_version": "1.0", "doc_id": "1", "meta": {"doc_id": "1", "doc_name": "1", "doc_type": null, "doc_subtype": null, "case_type": null, "doc_code": null, "doc_number": null, "year": null, "title": "Tổ chức các cơ quan Lao động địa phương liên khu và tỉnh", "subject": null, "issue_date": "1950-03-27", "issuing_body": null, "court_level": null, "jurisdiction": null, "precedent_number": null, "language": "vi"}, "stats": {"num_pages": 1, "num_sections": 1, "num_paragraphs": 1, "num_sentences": 6, "char_len": 1149, "text_hash": "6bba1a9f1ebb47741d63ffa6b2d29313"}, "sections": [{"section_id": "1#sec_00_header", "index": 0, "kind": "header", "label": null, "page_start": 1, "page_end": 1, "char_start": 0, "char_end": 1149, "paragraph_ids": ["1#par_0000"]}], "paragraphs": [{"paragraph_id": "1#par_0000", "index": 0, "section_id": "1#sec_00_header", "section_kind": "header", "page": 1, "char_start": 0, "char_end": 1150, "text": "BỘ LAO ĐỘNG CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập – Tự do – Hạnh phúc Số: 24/LĐ-NĐ Hà Nội, ngày 27 tháng 03 năm 1950 NGHỊ ĐỊNH BỘ TRƯỞNG BỘ LAO ĐỘNG Tổ chức các cơ quan Lao động địa phương liên khu và tỉnh BỘ TRƯỞNG BỘ LAO ĐỘNG Chiểu Sắc lệnh số 226/SL ngày 28/11/1946 tổ chức Bộ Lao động trong Chính phủ hiện thời; Chiểu Sắc lệnh số 169/SL ngày 14/04/1948 tổ chức các cơ quan Lao động địa phương; Chiểu Nghị định số 20 ngày 15/05/1948 của Bộ trưởng Bộ Lao động ấn định nhiệm vụ của các Sở và Ty Lao động; NGHỊ ĐỊNH: Điều 1. -Kể từ ngày ký nghị định này, các Sở Lao động sẽ gọi là \"Khu Lao động\", do một ông Giám đốc Khu Lao động điều khiển. Điều 2. -Ở tỉnh nào xét ra quan hệ nhiều về phương diện lao động sẽ đặt một Ty Lao động, do một Trưởng ty Lao động điều khiển. Ở những tỉnh nào kế cận với nhau mà vấn đề lao động ít quan hệ hơn thì đặt một Ty Lao động liên tỉnh. Ở các tỉnh khác công việc lao động sẽ do Ủy ban Kháng chiến hành chính tỉnh ấy trực tiếp đảm nhận. Điều 3. -Ông Đổng lý Văn phòng Bộ Lao động, các ông Chủ tịch UBKCHC các liên khu và các ông Giám đốc Khu Lao động chiểu nghị định thi hành./. BỘ TRƯỞNG (Đã ký) Nguyễn Văn Tạo", "kind": "text", "marker": null, "sentence_ids": ["1#sen_0000", "1#sen_0001", "1#sen_0002", "1#sen_0003", "1#sen_0004", "1#sen_0005"]}], "sentences": [{"sentence_id": "1#sen_0000", "paragraph_id": "1#par_0000", "section_id": "1#sec_00_header", "section_kind": "header", "page": 1, "index_in_paragraph": 0, "global_index": 0, "char_start": 0, "char_end": 646, "text": "BỘ LAO ĐỘNG CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM Độc lập – Tự do – Hạnh phúc Số: 24/LĐ-NĐ Hà Nội, ngày 27 tháng 03 năm 1950 NGHỊ ĐỊNH BỘ TRƯỞNG BỘ LAO ĐỘNG Tổ chức các cơ quan Lao động địa phương liên khu và tỉnh BỘ TRƯỞNG BỘ LAO ĐỘNG Chiểu Sắc lệnh số 226/SL ngày 28/11/1946 tổ chức Bộ Lao động trong Chính phủ hiện thời; Chiểu Sắc lệnh số 169/SL ngày 14/04/1948 tổ chức các cơ quan Lao động địa phương; Chiểu Nghị định số 20 ngày 15/05/1948 của Bộ trưởng Bộ Lao động ấn định nhiệm vụ của các Sở và Ty Lao động; NGHỊ ĐỊNH: Điều 1. -Kể từ ngày ký nghị định này, các Sở Lao động sẽ gọi là \"Khu Lao động\", do một ông Giám đốc Khu Lao động điều khiển."}, {"sentence_id": "1#sen_0001", "paragraph_id": "1#par_0000", "section_id": "1#sec_00_header", "section_kind": "header", "page": 1, "index_in_paragraph": 1, "global_index": 1, "char_start": 647, "char_end": 773, "text": "Điều 2. -Ở tỉnh nào xét ra quan hệ nhiều về phương diện lao động sẽ đặt một Ty Lao động, do một Trưởng ty Lao động điều khiển."}, {"sentence_id": "1#sen_0002", "paragraph_id": "1#par_0000", "section_id": "1#sec_00_header", "section_kind": "header", "page": 1, "index_in_paragraph": 2, "global_index": 2, "char_start": 774, "char_end": 875, "text": "Ở những tỉnh nào kế cận với nhau mà vấn đề lao động ít quan hệ hơn thì đặt một Ty Lao động liên tỉnh."}, {"sentence_id": "1#sen_0003", "paragraph_id": "1#par_0000", "section_id": "1#sec_00_header", "section_kind": "header", "page": 1, "index_in_paragraph": 3, "global_index": 3, "char_start": 876, "char_end": 974, "text": "Ở các tỉnh khác công việc lao động sẽ do Ủy ban Kháng chiến hành chính tỉnh ấy trực tiếp đảm nhận."}, {"sentence_id": "1#sen_0004", "paragraph_id": "1#par_0000", "section_id": "1#sec_00_header", "section_kind": "header", "page": 1, "index_in_paragraph": 4, "global_index": 4, "char_start": 975, "char_end": 1116, "text": "Điều 3. -Ông Đổng lý Văn phòng Bộ Lao động, các ông Chủ tịch UBKCHC các liên khu và các ông Giám đốc Khu Lao động chiểu nghị định thi hành./."}, {"sentence_id": "1#sen_0005", "paragraph_id": "1#par_0000", "section_id": "1#sec_00_header", "section_kind": "header", "page": 1, "index_in_paragraph": 5, "global_index": 5, "char_start": 1117, "char_end": 1149, "text": "BỘ TRƯỞNG (Đã ký) Nguyễn Văn Tạo"}]}
- **extracted_json**:
  > {"doc_id": "1", "text_hash": "6bba1a9f1ebb47741d63ffa6b2d29313", "char_len": 1149, "entities": [{"tag": "DATE", "text": "28/11/1946", "start": 263, "end": 273}, {"tag": "DATE", "text": "14/04/1948", "start": 351, "end": 361}, {"tag": "DATE", "text": "15/05/1948", "start": 430, "end": 440}, {"tag": "ARTICLE", "text": "Điều 1", "start": 522, "end": 528}, {"tag": "ARTICLE", "text": "Điều 2", "start": 647, "end": 653}, {"tag": "ARTICLE", "text": "Điều 3", "start": 975, "end": 981}], "relations": [], "statute_refs": [{"article": 1, "clause": null, "point": null, "code": null, "year": null, "span": [522, 528]}, {"article": 2, "clause": null, "point": null, "code": null, "year": null, "span": [647, 653]}, {"article": 3, "clause": null, "point": null, "code": null, "year": null, "span": [975, 981]}]}
- **file_paths_json**: `nan`