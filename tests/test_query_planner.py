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


class CombinedPlanningLLM:
    def __init__(self):
        self.calls = 0

    def call_llm_json(self, **kwargs):
        self.calls += 1
        return {
            "understanding": {
                "domain": "doanh nghiệp",
                "intent": "condition_question",
                "legal_entities": ["doanh nghiệp nhỏ và vừa"],
            },
            "plan": {
                "queries": [
                    {
                        "query_type": "legal_rewrite",
                        "text": "Điều kiện doanh nghiệp nhỏ và vừa được hỗ trợ",
                    },
                    {
                        "query_type": "keyword",
                        "text": "doanh nghiệp nhỏ vừa tiêu chí hỗ trợ",
                    },
                ],
                "filters": {"is_current": True},
            },
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


def test_combined_planning_uses_one_llm_call():
    llm = CombinedPlanningLLM()

    result = QueryPlannerAgent(llm).run_combined(
        "Doanh nghiệp nhỏ và vừa được hỗ trợ khi nào?"
    )

    assert llm.calls == 1
    assert result.understanding.intent == "condition_question"
    assert len(result.plan.queries) == 3
