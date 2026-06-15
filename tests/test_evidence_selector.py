from src.agents.evidence_selector import EvidenceSelectorAgent
from src.schema.agent_schemas import Evidence


class FakeLLM:
    def call_llm_json(self, **kwargs):
        return {
            "selected": [
                {
                    "unit_id": "u2",
                    "role": "main",
                    "reason": "Direct evidence",
                    "supported_claims": ["Condition"],
                },
                {
                    "unit_id": "unknown",
                    "role": "supporting",
                    "reason": "Invalid",
                    "supported_claims": [],
                },
            ],
            "rejected": [],
        }


def test_evidence_selector_validates_ids():
    candidates = [
        Evidence(unit_id="u1", chunk_id="c1", text="One"),
        Evidence(unit_id="u2", chunk_id="c2", text="Two"),
    ]
    result = EvidenceSelectorAgent(FakeLLM()).run("Question", candidates)

    assert [item.unit_id for item in result.selected] == ["u2"]


def test_get_selected_evidence_keeps_multiple_articles_per_document():
    candidates = [
        Evidence(
            unit_id="u1",
            chunk_id="c1",
            text="Document one",
            final_score=1.0,
            metadata={"doc_code": "01/2020/QH14", "article": "Dieu 1"},
        ),
        Evidence(
            unit_id="u2",
            chunk_id="c2",
            text="Document two, lower",
            final_score=1.5,
            metadata={"doc_code": "02/2020/QH14", "article": "Dieu 2"},
        ),
        Evidence(
            unit_id="u3",
            chunk_id="c3",
            text="Document two, higher",
            final_score=1.8,
            metadata={"doc_code": "02/2020/QH14", "article": "Dieu 3"},
        ),
    ]

    selected = EvidenceSelectorAgent(FakeLLM()).get_selected_evidence(
        candidates
    )

    assert [item.chunk_id for item in selected] == ["c3", "c2", "c1"]


def test_selector_keeps_clear_top_score():
    candidates = [
        Evidence(unit_id="top", text="Top", final_score=2.0),
        Evidence(unit_id="u2", text="Second", final_score=1.8),
    ]

    result = EvidenceSelectorAgent(FakeLLM()).run("Question", candidates)

    assert result.selected[0].unit_id == "top"
