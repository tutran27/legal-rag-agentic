import json

from src.agents.verifier import VerificationAgent
from src.schema.agent_schemas import AnswerDraft, Evidence


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.query = None

    def call_llm_json(self, **kwargs):
        self.query = kwargs["query"]
        return self.response


def make_evidence():
    return [
        Evidence(
            unit_id="u1",
            text="Nội dung đầy đủ.",
            metadata={
                "doc_code": "04/2017/QH14",
                "article": "Điều 5",
                "content_text": "Việc hỗ trợ phải công khai, minh bạch.",
            },
        )
    ]


def test_verifier_forces_failure_when_any_error_exists():
    llm = FakeLLM(
        {
            "passed": True,
            "unsupported_claims": ["Doanh nghiệp luôn được hỗ trợ."],
            "missing_citations": [],
            "extra_citations": [],
            "revision_instruction": None,
        }
    )

    report = VerificationAgent(llm).run(
        question="Nguyên tắc hỗ trợ là gì?",
        answer=AnswerDraft(answer="Doanh nghiệp luôn được hỗ trợ."),
        evidence=make_evidence(),
    )

    assert report.passed is False
    assert report.revision_instruction


def test_verifier_passes_only_when_all_error_lists_are_empty():
    llm = FakeLLM(
        {
            "passed": False,
            "unsupported_claims": [],
            "missing_citations": [],
            "extra_citations": [],
            "revision_instruction": "Không cần sửa.",
        }
    )

    report = VerificationAgent(llm).run(
        question="Nguyên tắc hỗ trợ là gì?",
        answer=AnswerDraft(
            answer="Việc hỗ trợ phải công khai, minh bạch."
        ),
        evidence=make_evidence(),
    )

    payload = json.loads(llm.query)
    assert report.passed is True
    assert report.revision_instruction is None
    assert payload["evidence"][0]["text"].endswith("minh bạch.")


def test_verifier_ignores_missing_citations_when_claim_is_supported():
    llm = FakeLLM(
        {
            "passed": False,
            "unsupported_claims": [],
            "missing_citations": [
                "Nội dung thuộc một điều khác trong selected evidence."
            ],
            "extra_citations": [],
            "revision_instruction": "Bổ sung trích dẫn.",
        }
    )

    report = VerificationAgent(llm).run(
        question="Nguyên tắc ưu tiên hỗ trợ là gì?",
        answer=AnswerDraft(answer="Doanh nghiệp nộp hồ sơ trước được hỗ trợ trước."),
        evidence=make_evidence(),
    )

    assert report.passed is True
    assert report.missing_citations == []
    assert report.revision_instruction is None


def test_verifier_ignores_extra_citation_classification():
    llm = FakeLLM(
        {
            "passed": False,
            "unsupported_claims": [],
            "missing_citations": [],
            "extra_citations": ["Không cần kiểm tra trích dẫn."],
            "revision_instruction": "Bỏ trích dẫn.",
        }
    )

    report = VerificationAgent(llm).run(
        question="Nguyên tắc hỗ trợ là gì?",
        answer=AnswerDraft(answer="Việc hỗ trợ phải công khai, minh bạch."),
        evidence=make_evidence(),
    )

    assert report.passed is True
    assert report.extra_citations == []
    assert report.revision_instruction is None


def test_verifier_rejects_empty_evidence_without_calling_llm():
    llm = FakeLLM({})

    report = VerificationAgent(llm).run(
        question="Nguyên tắc hỗ trợ là gì?",
        answer=AnswerDraft(answer="Một câu trả lời."),
        evidence=[],
    )

    assert report.passed is False
    assert llm.query is None
