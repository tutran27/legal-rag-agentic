from src.agents.sufficiency_checker import SufficiencyCheckerAgent
from src.schema.agent_schemas import (
    Evidence,
    EvidenceSelectionResult,
    LegalUnderstanding,
    SelectedEvidence,
)


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.query = None

    def call_llm_json(self, **kwargs):
        self.query = kwargs["query"]
        return self.response


def make_inputs():
    selection = EvidenceSelectionResult(
        selected=[
            SelectedEvidence(
                unit_id="u1",
                role="main",
                reason="Quy định đối tượng",
            )
        ]
    )
    evidence = [
        Evidence(
            unit_id="u1",
            text="Quy định đối tượng được hỗ trợ.",
            metadata={"doc_code": "04/2017/QH14", "article": "Điều 5"},
        )
    ]
    return selection, evidence


def test_sufficiency_checker_clears_retry_fields_when_sufficient():
    selection, evidence = make_inputs()
    agent = SufficiencyCheckerAgent(
        FakeLLM(
            {
                "is_sufficient": True,
                "reason": "Đã đủ căn cứ.",
                "missing_evidence": ["Không dùng"],
                "next_queries": ["Không dùng"],
            }
        )
    )

    report = agent.run(
        "Điều kiện hỗ trợ?",
        LegalUnderstanding(sub_questions=["Điều kiện"]),
        selection,
        evidence,
    )

    assert report.is_sufficient is True
    assert report.missing_evidence == []
    assert report.next_queries == []


def test_sufficiency_checker_builds_targeted_queries_when_missing():
    selection, evidence = make_inputs()
    agent = SufficiencyCheckerAgent(
        FakeLLM(
            {
                "is_sufficient": False,
                "reason": "Thiếu điều kiện cụ thể.",
                "missing_evidence": ["Điều kiện nhận hỗ trợ"],
                "next_queries": [],
            }
        )
    )

    report = agent.run(
        "Điều kiện hỗ trợ?",
        LegalUnderstanding(),
        selection,
        evidence,
    )

    assert report.is_sufficient is False
    assert report.next_queries == ["Điều kiện nhận hỗ trợ"]


def test_sufficiency_checker_handles_empty_evidence_without_llm():
    agent = SufficiencyCheckerAgent(FakeLLM({}))

    report = agent.run(
        "Điều kiện hỗ trợ?",
        LegalUnderstanding(),
        EvidenceSelectionResult(selected=[]),
        [],
    )

    assert report.is_sufficient is False
    assert report.next_queries == ["Điều kiện hỗ trợ?"]


def test_sufficiency_checker_sends_full_evidence_text():
    selection, evidence = make_inputs()
    evidence[0].text = "A" * 2000 + "END"
    llm = FakeLLM(
        {
            "is_sufficient": True,
            "reason": "Đủ căn cứ.",
            "missing_evidence": [],
            "next_queries": [],
        }
    )

    SufficiencyCheckerAgent(llm).run(
        "Điều kiện hỗ trợ?",
        LegalUnderstanding(),
        selection,
        evidence,
    )

    assert "END" in llm.query
