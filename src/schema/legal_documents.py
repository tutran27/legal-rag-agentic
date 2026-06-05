from pydantic import BaseModel
from typing import Optional


class LegalDocument(BaseModel):
    doc_id: str                                                 # ID của văn bản     VD: vbpl_12345
    doc_code: Optional[str]                                     # Mã văn bản         VD: 123/2023/PL-UBTVQH15
    doc_type: Optional[str]                                     # Loại văn bản       VD: Luật, Nghị định, Thông tư
    doc_title_raw: Optional[str]                                # Tiêu đề gốc        VD: Luật Đầu tư 2020
    doc_title_submission: Optional[str]                         # Tiêu đề đã xử lý   VD: Luật Đầu tư

    issuer: Optional[str] = None                                # Cơ quan ban hành   VD: Quốc hội, Chính phủ
    issued_date: Optional[str] = None                           # Ngày ban hành      VD: 2023-01-01
    effective_date: Optional[str] = None                        # Ngày hiệu lực      VD: 2023-01-01
    expired_date: Optional[str] = None                          # Ngày hết hiệu lực  VD: 2024-01-01
    status: Optional[str] = None                                # Trạng thái         VD: Còn hiệu lực, Hết hiệu lực

    source: str                                                 # Nguồn              VD: vbpl.vn, pháp luật Việt Nam
    source_url: Optional[str] = None                            # URL nguồn          VD: https://vbpl.vn/vbpl/123

    full_text: Optional[str] = None                             # Văn bản đầy đủ     VD: Văn bản đầy đủ của văn bản