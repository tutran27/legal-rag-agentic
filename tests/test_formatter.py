import pytest

from src.agents.formatter import SubmissionFormatterAgent
from src.schema.agent_schemas import (
    AnswerDraft,
    Evidence,
    VerificationReport,
)


def test_formatter_builds_unique_references():
    evidence = [
        Evidence(
            unit_id="u1",
            text="Nội dung 1",
            metadata={
                "doc_code": "04/2017/QH14",
                "article": "Điều 5",
            },
        ),
        Evidence(
            unit_id="u2",
            text="Nội dung 2",
            doc_code="04/2017/QH14",
            article="Điều 5",
        ),
        Evidence(
            unit_id="u3",
            text="Nội dung 3",
            doc_code="80/2021/NĐ-CP",
            article="Điều 4",
        ),
    ]

    result = SubmissionFormatterAgent().run(
        question_id=1,
        question="Điều kiện hỗ trợ là gì?",
        answer=AnswerDraft(answer="  Câu trả lời.  "),
        evidence=evidence,
        verification=VerificationReport(passed=True),
    )

    assert result.answer == "Câu trả lời."
    assert result.relevant_docs == [
        "04/2017/QH14",
        "80/2021/NĐ-CP",
    ]
    assert result.relevant_articles == [
        "04/2017/QH14 - Điều 5",
        "80/2021/NĐ-CP - Điều 4",
    ]


def test_formatter_rejects_unverified_answer():
    with pytest.raises(ValueError):
        SubmissionFormatterAgent().run(
            question_id=1,
            question="Câu hỏi",
            answer=AnswerDraft(answer="Câu trả lời"),
            evidence=[],
            verification=VerificationReport(passed=False),
        )
