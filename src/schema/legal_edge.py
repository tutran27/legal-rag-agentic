from pydantic import BaseModel


class LegalEdge(BaseModel):
    source_doc_id: str                                  # Văn bản nguồn trong quan hệ        VD: Nghị định 80/2021/NĐ-CP
    target_doc_id: str                                  # Văn bản đích trong quan hệ          VD: Luật 04/2017/QH14
    relation_type: str                                  # Loại quan hệ                       VD: "căn cứ", "sửa đổi", "bổ sung".. Dùng english

    source_doc_code: str | None = None                  # Mã văn bản nguồn                   VD: 80/2021/NĐ-CP
    target_doc_code: str | None = None                  # Mã văn bản đích                    VD: 04/2017/QH14

    evidence_text: str | None = None                    # Văn bảnEvidence                    VD: Nghị định này quy định chi tiết và hướng dẫn thi hành một số điều của Luật Hỗ trợ doanh nghiệp nhỏ và vừa.
    source_url: str | None = None                       # URL nguồn                          VD: http://vbpl.vn/TW/Pages/vbpq-toanvan.aspx?ItemID=18562