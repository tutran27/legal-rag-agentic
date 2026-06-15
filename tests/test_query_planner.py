from src.agents.query_planner import QueryPlannerAgent, _tokens
from src.schema.agent_schemas import LegalUnderstanding


class DuplicateQueryLLM:
    def call_llm_json(self, **kwargs):
        return {
            "queries": [
                {
                    "query_type": "legal_rewrite",
                    "text": "Điều kiện hỗ trợ doanh nghiệp nhỏ và vừa?",
                },
                {
                    "query_type": "keyword",
                    "text": "điều kiện hỗ trợ doanh nghiệp nhỏ và vừa",
                },
            ],
            "filters": {"is_current": True},
        }


def test_query_planner_replaces_repetitive_queries():
    question = "Doanh nghiệp nhỏ và vừa cần điều kiện nào để được hỗ trợ?"
    plan = QueryPlannerAgent(DuplicateQueryLLM()).run(
        question,
        LegalUnderstanding(
            domain="doanh nghiệp nhỏ và vừa",
            intent="condition_question",
        ),
    )

    assert [query.query_type for query in plan.queries] == [
        "original",
        "legal_rewrite",
        "keyword",
    ]
    assert "?" not in plan.queries[1].text
    assert len(_tokens(plan.queries[1].text) - _tokens(question)) >= 1
    assert len(_tokens(plan.queries[2].text) - _tokens(question)) >= 1
    assert plan.queries[1].text != plan.queries[2].text
