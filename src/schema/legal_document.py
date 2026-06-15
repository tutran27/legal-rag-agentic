from pydantic import BaseModel


class LegalDocument(BaseModel):
    doc_id: str
    source: str
    doc_code: str | None = None
    doc_type: str | None = None
    doc_title_raw: str | None = None
    doc_title_submission: str | None = None
    issuer: str | None = None
    issued_date: str | None = None
    effective_date: str | None = None
    expired_date: str | None = None
    status: str | None = None
    source_url: str | None = None
    full_text: str | None = None
