import pytest

from src.schema.agent_schemas import Evidence, SubmissionItem
from src.submission.validate_results import validate_submission_item


def make_evidence():
    return [
        Evidence(
            unit_id="u1",
            text="Nội dung",
            metadata={
                "doc_code": "04/2017/QH14",
                "article": "Điều 5",
            },
        )
    ]


def test_validator_accepts_selected_evidence_citations():
    item = SubmissionItem(
        id=1,
        question="Câu hỏi",
        answer="Câu trả lời",
        relevant_docs=["04/2017/QH14"],
        relevant_articles=["04/2017/QH14 - Điều 5"],
    )

    validate_submission_item(item, make_evidence())


def test_validator_rejects_unknown_citation():
    item = SubmissionItem(
        id=1,
        question="Câu hỏi",
        answer="Câu trả lời",
        relevant_docs=["Không tồn tại"],
        relevant_articles=[],
    )

    with pytest.raises(ValueError, match="Citation không thuộc"):
        validate_submission_item(item, make_evidence())
