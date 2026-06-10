from src.agents.query_planner import QueryPlannerAgent
from src.schema.agent_schemas import LegalUnderstanding


class FakeLLM:
    def call_llm_json(self, **kwargs):
        return {
            "queries": [
                {
                    "query_type": "legal_rewrite",
                    "text": "chính sách hỗ trợ cơ sở ươm tạo về thuế đất",
                }
            ],
            "filters": {
                "domains": ["doanh nghiệp"],
                "is_current": True,
            },
            "retrieval": {
                "use_bm25": False,
                "top_k_dense": 50,
                "top_k_sparse": 50,
                "top_k_colbert": 20,
            },
        }


def test_query_planner_adds_original_query():
    planner = QueryPlannerAgent(FakeLLM())
    plan = planner.run(
        "Cơ sở ươm tạo được hỗ trợ gì về thuế và đất đai?",
        LegalUnderstanding(domain="doanh nghiệp"),
    )

    assert plan.queries[0].query_type == "original"
    assert plan.filters.is_current is True
    assert plan.retrieval.use_colbert is True
