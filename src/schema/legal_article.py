from pydantic import BaseModel
from typing import Optional


class LegalArticle(BaseModel):
    article_id: str                                    # ID của điều

    doc_id: Optional[str]                                       # ID của văn bản     VD: vbpl_12345
    doc_code: Optional[str]                                     # Mã văn bản         VD: 123/2023/PL-UBTVQH15
    doc_type: Optional[str]                                     # Loại văn bản       VD: Luật, Nghị định, Thông tư
    doc_title_submission: Optional[str]                         # Tiêu đề đã xử lý   VD: Luật Đầu tư

    article_id: str                                                # Số điều
    article_title: Optional[str]                                # Tiêu đề điều       VD: Điều 1.1.LQ.1. Phạm vi điều chỉnh

    topic_title: Optional[str] = None                           # Chủ đề     VD: An ninh quốc gia
    subject_title: Optional[str] = None                         # Mục        VD: An ninh quốc gia
    chapter_title: Optional[str] = None                         # Chương     VD: Chương I - NHỮNG QUY ĐỊNH CHUNG

    content_text: str                                           # Nội dung điều      VD: Luật này quy định về chính sách an ninh quốc gia;
    source_note_text: Optional[str] = None                      # Ghi chú nguồn      VD: (Điều 1 Luật số 32/2004/QH11 An ninh Quốc gia ngày 03/12/2004 của Quốc hội, có hiệu lực thi hành kể từ ngày 01/07/2005 )
    source_url: Optional[str] = None                            # URL nguồn          VD: http://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=18562#Chuong_I_Dieu_1

    effective_status: Optional[str] = None                      # Trạng thái hiệu lực