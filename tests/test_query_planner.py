from src.agents.query_planner import QueryPlannerAgent, _tokens


class PlanOnlyLLM:
    def __init__(self, plan):
        self.plan = plan
        self.calls = 0
        self.system_prompt = None

    def call_llm_json(self, **kwargs):
        self.calls += 1
        self.system_prompt = kwargs["system_prompt"]
        return {"plan": self.plan}


def test_query_planner_returns_only_original_and_rewrite():
    question = (
        "Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng "
        "thì bị xử lý như thế nào?"
    )
    llm = PlanOnlyLLM(
        {
            "queries": [
                {
                    "query_type": "original",
                    "text": question,
                    "reason": "Câu hỏi gốc",
                },
                {
                    "query_type": "legal_rewrite",
                    "text": (
                        "công ty giữ bản chính bằng cấp của người lao động "
                        "khi giao kết hợp đồng lao động bị xử lý"
                    ),
                    "reason": "Viết lại pháp lý bám sát câu hỏi",
                },
            ],
            "filters": {"is_current": True},
        }
    )

    result = QueryPlannerAgent(llm).run(question)

    assert llm.calls == 1
    assert "Không trả phần understanding" in llm.system_prompt
    assert [query.query_type for query in result.plan.queries] == [
        "original",
        "legal_rewrite",
    ]
    assert "keyword" not in result.model_dump_json()
    assert result.understanding.intent is None


def test_query_planner_replaces_unaccented_rewrite():
    question = "Doanh nghiệp nhỏ và vừa được hỗ trợ khi nào?"
    llm = PlanOnlyLLM(
        {
            "queries": [
                {
                    "query_type": "legal_rewrite",
                    "text": "dieu kien ho tro doanh nghiep nho va vua",
                }
            ],
            "filters": {"is_current": True},
        }
    )

    result = QueryPlannerAgent(llm).run(question)
    rewrite = result.plan.queries[1].text

    assert "dieu kien" not in rewrite
    assert "doanh nghiep" not in rewrite
    assert "điều kiện" in rewrite
    assert len(_tokens(rewrite)) >= len(_tokens(question))


def test_query_planner_normalizes_taxonomy_to_vietnamese_lowercase():
    llm = PlanOnlyLLM(
        {
            "queries": [
                {
                    "query_type": "legal_rewrite",
                    "text": "điều kiện doanh nghiệp nhỏ và vừa được hỗ trợ",
                }
            ],
            "filters": {
                "domains": ["doanh-nghiep"],
                "sectors": ["ke-hoach-va-dau-tu"],
                "is_current": True,
            },
        }
    )

    result = QueryPlannerAgent(llm).run("Câu hỏi")

    assert result.plan.filters.domains == ["doanh nghiệp"]
    assert result.plan.filters.sectors == ["kế hoạch và đầu tư"]
