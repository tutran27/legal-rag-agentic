import json

import pytest

from src.schema.agent_schemas import Evidence, SubmissionItem
from src.submission.validate_results import (
    load_and_validate_results,
    validate_submission_item,
)


def make_evidence():
    return [
        Evidence(
            unit_id="u1",
            text="Noi dung",
            metadata={
                "doc_code": "04/2017/QH14",
                "doc_title_submission": "Luat 04/2017/QH14 Ho tro doanh nghiep nho va vua",
                "article": "Dieu 5",
            },
        )
    ]


def test_validator_accepts_selected_evidence_citations():
    item = SubmissionItem(
        id=1,
        question="Cau hoi",
        answer="Cau tra loi",
        relevant_docs=[
            "04/2017/QH14|Luat 04/2017/QH14 Ho tro doanh nghiep nho va vua"
        ],
        relevant_articles=[
            "04/2017/QH14|Luat 04/2017/QH14 Ho tro doanh nghiep nho va vua|Dieu 5"
        ],
    )

    validate_submission_item(item, make_evidence())


def test_validator_rejects_unknown_citation():
    item = SubmissionItem(
        id=1,
        question="Cau hoi",
        answer="Cau tra loi",
        relevant_docs=["Khong ton tai"],
        relevant_articles=[],
    )

    with pytest.raises(ValueError, match="Citation"):
        validate_submission_item(item, make_evidence())


def test_load_and_validate_results(tmp_path):
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "question": "Cau hoi",
                    "answer": "Cau tra loi",
                    "relevant_docs": [],
                    "relevant_articles": [],
                }
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    assert load_and_validate_results(path)[0].id == 1
