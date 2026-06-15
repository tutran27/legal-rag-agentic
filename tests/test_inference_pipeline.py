from src.pipeline import inference_pipeline as pipeline_module
from src.pipeline.inference_pipeline import InferencePipeline
from src.schema.agent_schemas import (
    AnswerDraft,
    Evidence,
    EvidenceAssessment,
    EvidenceSelectionResult,
    LegalUnderstanding,
    PlanningResult,
    QueryPlan,
    RetrievalPlan,
    SearchQuery,
    SelectedEvidence,
    SufficiencyReport,
    VerificationReport,
)


def test_pipeline_runs_with_injected_dependencies(monkeypatch):
    evidence = Evidence(
        unit_id="doc-1::article-1",
        text="Nội dung hỗ trợ doanh nghiệp.",
        final_score=1.0,
        metadata={"doc_code": "04/2017/QH14", "article": "Điều 5"},
    )
    planning = PlanningResult(
        understanding=LegalUnderstanding(intent="tra cứu"),
        plan=QueryPlan(
            queries=[
                SearchQuery(
                    query_type="original",
                    text="Câu hỏi",
                )
            ]
        ),
    )
    assessment = EvidenceAssessment(
        selection=EvidenceSelectionResult(
            selected=[
                SelectedEvidence(
                    unit_id=evidence.unit_id,
                    reason="Liên quan trực tiếp",
                )
            ]
        ),
        sufficiency=SufficiencyReport(
            is_sufficient=True,
            reason="Đủ căn cứ",
        ),
    )

    class FakePlanner:
        def __init__(self, llm):
            pass

        def run_combined(self, question):
            return planning

    class FakeSelector:
        def __init__(self, llm):
            pass

        def get_selected_evidence(self, candidates):
            return candidates

        def run_with_sufficiency(self, **kwargs):
            return assessment

    class FakeReasoner:
        def __init__(self, llm):
            pass

        def run(self, **kwargs):
            return AnswerDraft(answer="Doanh nghiệp đáp ứng điều kiện hỗ trợ.")

    class FakeVerifier:
        def __init__(self, llm):
            pass

        def run(self, **kwargs):
            return VerificationReport(passed=True)

    monkeypatch.setattr(pipeline_module, "QueryPlannerAgent", FakePlanner)
    monkeypatch.setattr(pipeline_module, "EvidenceSelectorAgent", FakeSelector)
    monkeypatch.setattr(pipeline_module, "ReasonerAgent", FakeReasoner)
    monkeypatch.setattr(pipeline_module, "VerificationAgent", FakeVerifier)

    pipeline = InferencePipeline(
        llm=object(),
        dense_model=object(),
        colbert_model=object(),
        cross_encoder=object(),
        preload_graph_index=False,
        verbose=False,
    )
    monkeypatch.setattr(
        pipeline,
        "_retrieve",
        lambda question, plan, latencies: [evidence],
    )
    monkeypatch.setattr(
        pipeline,
        "_expand",
        lambda question, plan, candidates, latencies: candidates,
    )
    monkeypatch.setattr(
        pipeline,
        "_rerank",
        lambda question, plan, candidates, latencies: candidates,
    )

    result = pipeline.run("Câu hỏi", question_id=7)

    assert result.submission.id == 7
    assert result.submission.relevant_docs == ["04/2017/QH14"]
    assert result.selected_evidence == [evidence]
    assert result.verification.passed is True


def test_expansion_gates_graph_and_context():
    strong = Evidence(
        unit_id="u1",
        text="Nội dung",
        vote_count=2,
        metadata={"part_count": 1},
    )
    weak = strong.model_copy(update={"vote_count": 1})
    split = strong.model_copy(
        update={"metadata": {"part_count": 2, "part_index": 0}}
    )

    assert InferencePipeline._needs_graph("Câu hỏi thông thường", [strong]) is False
    assert InferencePipeline._needs_graph("Văn bản hướng dẫn", [strong]) is True
    assert InferencePipeline._needs_graph("Câu hỏi thông thường", [weak]) is True
    assert InferencePipeline._needs_context([strong]) is False
    assert InferencePipeline._needs_context([split]) is True


def test_retrieve_passes_shared_client_to_exact(monkeypatch):
    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.verbose = False
    pipeline.dense_model = object()
    pipeline.qdrant_client = object()
    plan = QueryPlan(
        queries=[SearchQuery(query_type="original", text="Câu hỏi")],
        retrieval=RetrievalPlan(
            use_dense=False,
            use_sparse=True,
            use_bm25=True,
            use_exact=True,
            use_summary=False,
        ),
    )
    monkeypatch.setattr(
        pipeline_module,
        "hybrid_search_batch",
        lambda *args, **kwargs: [[]],
    )
    received = {}

    def fake_exact(*args, **kwargs):
        received.update(kwargs)
        return []

    monkeypatch.setattr(pipeline_module, "exact_search", fake_exact)

    pipeline._retrieve("Câu hỏi", plan, {})

    assert received["client"] is pipeline.qdrant_client
    assert "graph_path" not in received


def test_run_many_keeps_order_and_returns_item_error(monkeypatch):
    planning = PlanningResult(
        understanding=LegalUnderstanding(intent="lookup"),
        plan=QueryPlan(
            queries=[SearchQuery(query_type="original", text="Question")]
        ),
    )

    class FakePlanner:
        def __init__(self, llm):
            pass

        def run_combined(self, question):
            if question == "bad":
                raise ValueError("planning failed")
            return planning

    pipeline = InferencePipeline.__new__(InferencePipeline)
    pipeline.llm = object()
    monkeypatch.setattr(pipeline_module, "QueryPlannerAgent", FakePlanner)
    monkeypatch.setattr(
        pipeline,
        "_retrieve_many",
        lambda items: [[Evidence(unit_id="u", text="text")]],
    )
    monkeypatch.setattr(
        pipeline,
        "_complete",
        lambda question, question_id, *args: question_id,
    )

    results = pipeline.run_many([(1, "good"), (2, "bad")])

    assert results[0] == 1
    assert isinstance(results[1], ValueError)
