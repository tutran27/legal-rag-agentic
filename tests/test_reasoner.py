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


def test_reasoner_uses_selected_evidence_and_short_prompt():
    llm = FakeLLM(
        {
            "answer": "Doanh nghiep phai dap ung cac dieu kien ho tro theo noi dung duoc cung cap."
        }
    )
    evidence = [
        Evidence(
            unit_id="u1",
            text="Tieu de va noi dung day du.",
            final_score=1.5,
            metadata={
                "doc_code": "04/2017/QH14",
                "article": "Dieu 5",
                "content_text": "Noi dung phap ly chinh.",
            },
        )
    ]

    result = ReasonerAgent(llm).run(
        question="Dieu kien ho tro la gi?",
        understanding=LegalUnderstanding(),
        evidence=evidence,
    )

    payload = json.loads(llm.query)
    assert (
        result.answer
        == "Doanh nghiep phai dap ung cac dieu kien ho tro theo noi dung duoc cung cap."
    )
    assert payload["evidence"][0]["final_score"] == 1.5
    assert payload["evidence"][0]["text"] == "Noi dung phap ly chinh."
    assert "Doc selected evidence va tra loi dung trong tam cau hoi." in llm.system_prompt
    assert "Muc tieu toi da khoang 80 tu." in llm.system_prompt
    assert "Khong nhac den dieu luat" in llm.system_prompt
    assert llm.kwargs["max_new_tokens"] == 224


def test_reasoner_stops_when_evidence_is_empty():
    llm = FakeLLM({"answer": "Khong duoc goi."})

    result = ReasonerAgent(llm).run(
        question="Dieu kien ho tro la gi?",
        understanding=LegalUnderstanding(),
        evidence=[],
    )

    assert "Chua co du can cu phap ly" in result.answer
    assert llm.query is None
