# Pháp điển articles

Path:
data/raw/phapdien/articles_train.parquet

Rows:
64,464

Important columns:
- topic_title: chủ đề (Ví dụ thực tế: `An ninh quốc gia`)
- subject_title: đề mục (Ví dụ thực tế: `An ninh quốc gia`)
- article_title: tiêu đề điều pháp điển (Ví dụ thực tế: `Điều 1.1.LQ.1. Phạm vi điều chỉnh`)
- source_note_text: chứa Điều gốc + mã văn bản (Ví dụ thực tế: `(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia ngày 03/12/2004 của Q...`)
- content_text: nội dung điều (Ví dụ thực tế: `Luật này quy định về chính sách an ninh quốc gia; nguyên tắc, nhiệm...`)
- source_url: URL nguồn (Ví dụ thực tế: `https://phapdien.moj.gov.vn/TraCuuPhapDien/ViewBoPD.aspx?obj=&demuc...`)

Decision:
- Dùng content_text làm text chính
- Dùng source_note_text để parse doc_code và Điều gốc
- Không dùng article_title trực tiếp làm Điều output

Full Sample Details:
- **subject_id**: `55323c64-e78f-4537-afcd-6a3c2af3c71d`
- **topic_id**: `c3b69131-2931-4f67-926e-b244e18e8081`
- **topic_number**: `1`
- **topic_title**: `An ninh quốc gia`
- **subject_number**: `1`
- **subject_title**: `An ninh quốc gia`
- **article_anchor**: `#0100100000000000100000100000000000000000`
- **article_title**: `Điều 1.1.LQ.1. Phạm vi điều chỉnh`
- **chapter_title**: `Chương I - NHỮNG QUY ĐỊNH CHUNG`
- **source_note_text**:
  > (Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia ngày 03/12/2004 của Quốc hội, có hiệu lực thi hành kể từ ngày 01/07/2005 )
- **source_links**: `[{'text': '(Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia ngày 03/12/2004 của Quốc hội, có hiệu lực thi hành kể từ ngày 01/07/2005 )', 'href': 'http://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=18562#Chuong_I_Dieu_1'}]`
- **related_note_text**:
  > (Điều này có nội dung liên quan đến Điều 1.12.LQ.11. Biện pháp, chế độ cảnh vệ đối với người giữ chức vụ, chức danh cấp cao của Đảng Cộng sản Việt Nam, Nhà nước Cộng hòa xã hội chủ nghĩa Việt Nam, Ủy ban Trung ương Mặt trận Tổ quốc Việt Nam ; Điều 1.11.LQ.1. Phạm vi điều chỉnh ; Điều 1.6.LQ.16. Nhiệm vụ và quyền hạn của Công an nhân dân )
- **content_text**:
  > Luật này quy định về chính sách an ninh quốc gia; nguyên tắc, nhiệm vụ, biện pháp bảo vệ an ninh quốc gia; quyền, nghĩa vụ, trách nhiệm của cơ quan, tổ chức, công dân trong bảo vệ an ninh quốc gia.
- **content_char_len**: `197`
- **content_word_count**: `42`
- **source_url**:
  > https://phapdien.moj.gov.vn/TraCuuPhapDien/ViewBoPD.aspx?obj=&demucid=55323c64-e78f-4537-afcd-6a3c2af3c71d&mapc=1#0100100000000000100000100000000000000000
- **scraped_at**: `2026-05-08T15:49:05+00:00`