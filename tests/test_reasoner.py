import json

from src.agents.reasoner import ReasonerAgent
from src.schema.agent_schemas import Evidence


class FakeLLM:
    def __init__(self, response):
        self.response = response
        self.query = None
        self.system_prompt = None

    def call_llm_json(self, **kwargs):
        self.query = kwargs["query"]
        self.system_prompt = kwargs["system_prompt"]
        return self.response


def test_reasoner_uses_evidence_and_metadata():
    llm = FakeLLM(
        {
            "answer": "Doanh nghiệp phải đáp ứng các điều kiện được quy định."
        }
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
        evidence=evidence,
    )

    payload = json.loads(llm.query)
    assert result.answer == "Doanh nghiệp phải đáp ứng các điều kiện được quy định."
    assert "understanding" not in payload
    assert payload["evidence"][0]["role"] == "main"
    assert payload["evidence"][0]["text"] == "Nội dung pháp lý chính."
    assert "lấy ra thông tin phù hợp trực tiếp" in llm.system_prompt
    assert "Ưu tiên điều kiện, đối tượng, ngoại lệ" in llm.system_prompt


def test_reasoner_stops_when_no_evidence():
    llm = FakeLLM({"answer": "Không được gọi."})

    result = ReasonerAgent(llm).run(
        question="Điều kiện hỗ trợ là gì?",
        evidence=[],
    )

    assert "Chưa có đủ căn cứ pháp lý" in result.answer
    assert llm.query is None
