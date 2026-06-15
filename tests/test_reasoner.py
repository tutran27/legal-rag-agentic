import json

from src.agents.reasoner import ReasonerAgent
from src.schema.agent_schemas import (
    Evidence,
    EvidenceSelectionResult,
    LegalUnderstanding,
    SelectedEvidence,
    SufficiencyReport,
)


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.query = None
        self.system_prompt = None

    def call_llm_json(self, **kwargs):
        self.query = kwargs["query"]
        self.system_prompt = kwargs["system_prompt"]
        return self.response


def test_reasoner_uses_selected_evidence_and_metadata():
    llm = FakeLLM(
        {
            "answer": "Doanh nghiệp phải đáp ứng các điều kiện được quy định."
        }
    )
    selection = EvidenceSelectionResult(
        selected=[
            SelectedEvidence(
                unit_id="u1",
                role="main",
                reason="Quy định trực tiếp.",
                supported_claims=["Điều kiện hỗ trợ"],
            )
        ]
    )
    evidence = [
        Evidence(
            unit_id="u1",
            text="Tiêu đề và nội dung đầy đủ.",
            metadata={
                "doc_code": "04/2017/QH14",
                "article": "Điều 5",
                "content_text": "Nội dung pháp lý chính.",
            },
        )
    ]

    result = ReasonerAgent(llm).run(
        question="Điều kiện hỗ trợ là gì?",
        understanding=LegalUnderstanding(),
        selection=selection,
        evidence=evidence,
        sufficiency=SufficiencyReport(
            is_sufficient=True,
            reason="Có căn cứ liên quan.",
        ),
    )

    payload = json.loads(llm.query)
    assert result.answer == "Doanh nghiệp phải đáp ứng các điều kiện được quy định."
    assert payload["evidence"][0]["role"] == "main"
    assert payload["evidence"][0]["selection_reason"] == "Quy định trực tiếp."
    assert payload["evidence"][0]["text"] == "Nội dung pháp lý chính."
    assert "không được lược bỏ các chi tiết cần thiết" in llm.system_prompt
    assert "thứ tự ưu tiên, ngoại lệ, giới hạn và thủ tục" in llm.system_prompt


def test_reasoner_stops_when_evidence_is_insufficient():
    llm = FakeLLM({"answer": "Không được gọi."})

    result = ReasonerAgent(llm).run(
        question="Điều kiện hỗ trợ là gì?",
        understanding=LegalUnderstanding(),
        selection=EvidenceSelectionResult(selected=[]),
        evidence=[Evidence(unit_id="u1", text="Không liên quan.")],
        sufficiency=SufficiencyReport(
            is_sufficient=False,
            reason="Không liên quan.",
        ),
    )

    assert "Chưa có đủ căn cứ pháp lý" in result.answer
    assert llm.query is None
