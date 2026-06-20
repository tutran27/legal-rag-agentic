import json

from src.agents.reasoner import ReasonerAgent
from src.schema.agent_schemas import (
    Evidence,
    LegalUnderstanding,
)


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.query = None
        self.system_prompt = None
        self.kwargs = None

    def call_llm_json(self, **kwargs):
        self.kwargs = kwargs
        self.query = kwargs["query"]
        self.system_prompt = kwargs["system_prompt"]
        return self.response


def test_reasoner_uses_final_evidence_and_metadata():
    llm = FakeLLM(
        {
            "answer": "Doanh nghiệp phải đáp ứng các điều kiện được quy định."
        }
    )
    evidence = [
        Evidence(
            unit_id="u1",
            text="Tiêu đề và nội dung đầy đủ.",
            final_score=1.5,
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
        evidence=evidence,
    )

    payload = json.loads(llm.query)
    assert result.answer == "Doanh nghiệp phải đáp ứng các điều kiện được quy định."
    assert payload["evidence"][0]["final_score"] == 1.5
    assert payload["evidence"][0]["text"] == "Nội dung pháp lý chính."
    assert "lấy ra những thông tin phù hợp trực tiếp" in llm.system_prompt
    assert "Ưu tiên điều kiện, đối tượng, ngoại lệ và thủ tục" in llm.system_prompt
    assert "tối đa 120 từ" in llm.system_prompt
    assert "tối đa 4 gạch đầu dòng" in llm.system_prompt
    assert "không được bỏ sót điều kiện" in llm.system_prompt
    assert "ưu tiên trả lời đủ ý" in llm.system_prompt
    assert llm.kwargs["max_new_tokens"] == 192


def test_reasoner_stops_when_evidence_is_empty():
    llm = FakeLLM({"answer": "Không được gọi."})

    result = ReasonerAgent(llm).run(
        question="Điều kiện hỗ trợ là gì?",
        understanding=LegalUnderstanding(),
        evidence=[],
    )

    assert "Chưa có đủ căn cứ pháp lý" in result.answer
    assert llm.query is None
