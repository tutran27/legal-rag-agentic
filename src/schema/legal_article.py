from pydantic import BaseModel


class LegalArticle(BaseModel):
    article_id: str
    content_text: str
    doc_id: str | None = None
    doc_code: str | None = None
    doc_type: str | None = None
    doc_title_submission: str | None = None
    article_title: str | None = None
    topic_title: str | None = None
    subject_title: str | None = None
    chapter_title: str | None = None
    source_note_text: str | None = None
    source_url: str | None = None
    effective_status: str | None = None
