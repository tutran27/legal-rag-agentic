from src.schema.agent_schemas import (
    AnswerDraft,
    Evidence,
    SubmissionItem,
)


class SubmissionFormatterAgent:
    def run(
        self,
        question_id: int,
        question: str,
        answer: AnswerDraft,
        evidence: list[Evidence],
    ) -> SubmissionItem:
        relevant_docs = []
        relevant_articles = []

        for item in evidence:
            doc_code = item.doc_code or item.metadata.get("doc_code")
            doc_title = (
                item.doc_title_submission
                or item.metadata.get("doc_title_submission")
                or item.metadata.get("doc_name_for_submission")
            )
            article = item.article or item.metadata.get("article")

            if not doc_code or not doc_title:
                continue

            doc_citation = f"{doc_code}|{doc_title}"
            if doc_citation not in relevant_docs:
                relevant_docs.append(doc_citation)

            if article:
                citation = f"{doc_citation}|{article}"
                if citation not in relevant_articles:
                    relevant_articles.append(citation)

        return SubmissionItem(
            id=question_id,
            question=question,
            answer=answer.answer.strip(),
            relevant_docs=relevant_docs,
            relevant_articles=relevant_articles,
        )
