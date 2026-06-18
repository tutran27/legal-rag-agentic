from src.agents.query_planner import QueryPlannerAgent, _tokens


class DuplicateQueryLLM:
    def call_llm_json(self, **kwargs):
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
                        "text": "Điều kiện hỗ trợ doanh nghiệp nhỏ và vừa?",
                    },
                    {
                        "query_type": "keyword",
                        "text": "điều kiện hỗ trợ doanh nghiệp nhỏ và vừa",
                    },
                ],
                "filters": {"is_current": True},
            },
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
    result = QueryPlannerAgent(DuplicateQueryLLM()).run(question)
    plan = result.plan

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

    result = QueryPlannerAgent(llm).run(
        "Doanh nghiệp nhỏ và vừa được hỗ trợ khi nào?"
    )

    assert llm.calls == 1
    assert result.understanding.intent == "condition_question"
    assert len(result.plan.queries) == 3


def test_query_planner_normalizes_taxonomy_to_lowercase():
    class TaxonomyLLM:
        def call_llm_json(self, **kwargs):
            return {
                "understanding": {
                    "domain": "Doanh nghiệp",
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
                    "filters": {
                        "domains": ["Doanh nghiệp"],
                        "sectors": ["Kế hoạch và Đầu tư"],
                        "is_current": True,
                    },
                },
            }

    result = QueryPlannerAgent(TaxonomyLLM()).run("Câu hỏi")

    assert result.plan.filters.domains == ["doanh nghiệp"]
    assert result.plan.filters.sectors == ["kế hoạch và đầu tư"]


def test_query_planner_normalizes_taxonomy_slug_to_accented_text():
    class SlugTaxonomyLLM:
        def call_llm_json(self, **kwargs):
            return {
                "understanding": {
                    "domain": "doanh-nghiep",
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
                    "filters": {
                        "domains": ["doanh-nghiep"],
                        "sectors": ["ke-hoach-va-dau-tu"],
                        "is_current": True,
                    },
                },
            }

    result = QueryPlannerAgent(SlugTaxonomyLLM()).run("Câu hỏi")

    assert result.plan.filters.domains == ["doanh nghiệp"]
    assert result.plan.filters.sectors == ["kế hoạch và đầu tư"]


def test_query_planner_replaces_unaccented_queries():
    class UnaccentedLLM:
        def call_llm_json(self, **kwargs):
            return {
                "understanding": {
                    "domain": "doanh nghiep",
                    "intent": "condition_question",
                    "legal_entities": ["doanh nghiep nho va vua"],
                },
                "plan": {
                    "queries": [
                        {
                            "query_type": "legal_rewrite",
                            "text": "dieu kien ho tro doanh nghiep nho va vua",
                        },
                        {
                            "query_type": "keyword",
                            "text": "doanh nghiep nho va vua dieu kien ho tro",
                        },
                    ],
                    "filters": {"is_current": True},
                },
            }

    result = QueryPlannerAgent(UnaccentedLLM()).run(
        "Doanh nghiệp nhỏ và vừa được hỗ trợ khi nào?"
    )

    assert "doanh nghiep" not in result.plan.queries[1].text
    assert "dieu kien" not in result.plan.queries[2].text
    assert "điều kiện" in result.plan.queries[1].text
    assert "tiêu chí" in result.plan.queries[2].text
