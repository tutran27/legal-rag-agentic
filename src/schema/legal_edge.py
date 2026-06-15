from pydantic import BaseModel


class LegalEdge(BaseModel):
    source_doc_id: str
    target_doc_id: str
    relation_type: str
    source_doc_code: str | None = None
    target_doc_code: str | None = None
    evidence_text: str | None = None
    source_url: str | None = None
