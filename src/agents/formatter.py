from src.schema.agent_schemas import (
    AnswerDraft,
    Evidence,
    SubmissionItem,
    VerificationReport,
)


class SubmissionFormatterAgent:
    def run(
        self,
        question_id: int,
        question: str,
        answer: AnswerDraft,
        evidence: list[Evidence],
        verification: VerificationReport,
    ) -> SubmissionItem:
        if not verification.passed:
            raise ValueError("Câu trả lời chưa vượt qua bước verification.")

        relevant_docs = []
        relevant_articles = []

        for item in evidence:
            doc_code = item.doc_code or item.metadata.get("doc_code")
            article = item.article or item.metadata.get("article")

            if doc_code and doc_code not in relevant_docs:
                relevant_docs.append(str(doc_code))

            if article:
                citation = (
                    f"{doc_code} - {article}"
                    if doc_code
                    else str(article)
                )
                if citation not in relevant_articles:
                    relevant_articles.append(citation)

        return SubmissionItem(
            id=question_id,
            question=question,
            answer=answer.answer.strip(),
            relevant_docs=relevant_docs,
            relevant_articles=relevant_articles,
        )
