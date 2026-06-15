import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from qdrant_client import models
from sentence_transformers import CrossEncoder

from src.agents.legal_understanding import LegalUnderstandingAgent
from src.agents.formatter import SubmissionFormatterAgent
from src.agents.query_planner import QueryPlannerAgent
from src.agents.reasoner import ReasonerAgent
from src.agents.sufficiency_checker import SufficiencyCheckerAgent
from src.agents.verifier import VerificationAgent
from src.common.embedding import load_colbert_model, load_dense_model
from src.generation.llm_service import GroqLLMClient
from src.retrieval.colbert_reranker import colbert_rerank
from src.retrieval.context_expander import expand_context
from src.retrieval.cross_encoder_rerank import cross_encoder_rerank
from src.retrieval.exact_retriever import exact_search
from src.retrieval.fusion import rrf_fusion
from src.retrieval.graph_retriever import graph_search
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.summary_retriever import summary_search
from src.schema.agent_schemas import Evidence, RetrievalFilter
from src.agents.evidence_selector import EvidenceSelectorAgent
from src.submission.build_results import write_results
from src.submission.validate_results import validate_submission_item

DEFAULT_QUERY = (
    "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ "
    "theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?"
)
CROSS_ENCODER_MODEL = "Qwen/Qwen3-Reranker-0.6B"

def log_latency(name: str, started: float, latencies: dict[str, float]) -> None:
    elapsed = time.perf_counter() - started
    latencies[name] = elapsed
    print(f"[LATENCY] {name}: {elapsed:.3f}s")


def build_qdrant_filter(
    filters: RetrievalFilter,
    include_taxonomy: bool = True,
) -> models.Filter | None:
    conditions = []

    for key, values in (
        ("doc_code", filters.doc_codes),
        ("doc_type", filters.doc_types),
        ("domain", filters.domains if include_taxonomy else []),
        ("sector", filters.sectors if include_taxonomy else []),
    ):
        if values:
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=values),
                )
            )

    if filters.is_current is not None:
        conditions.append(
            models.FieldCondition(
                key="is_current",
                match=models.MatchValue(value=filters.is_current),
            )
        )
    return models.Filter(must=conditions) if conditions else None

def run_hybrid(queries, plan, dense_model, qdrant_filter):
    top_n = max(
        plan.top_k_bm25,
        plan.top_k_dense,
        plan.top_k_sparse,
    )
    results = []
    for index, search_query in enumerate(queries, start=1):
        print(f"Hybrid {index}/{len(queries)}: {search_query.text}")
        results.append(
            hybrid_search(
                query=search_query.text,
                dense_st=dense_model,
                colbert_model=None,
                cross_encoder=None,
                flt=qdrant_filter,
                top_n=top_n,
                rerank=False,
                use_dense=plan.use_dense,
                use_sparse=plan.use_bm25 or plan.use_sparse,
            )
        )
    return results


def print_results(candidates: list[Evidence]) -> None:
    print(f"\n=== KẾT QUẢ CUỐI ({len(candidates)}) ===")
    for rank, candidate in enumerate(candidates, start=1):
        metadata = candidate.metadata
        print(f"\n[{rank}] chunk={candidate.chunk_id} source={candidate.source}")
        print(
            f"scores: fusion={candidate.score:.6f} | "
            f"colbert={candidate.colbert_rerank_score or 0.0:.6f} | "
            f"colbert_norm={candidate.colbert_normalized_score or 0.0:.6f} | "
            f"cross_encoder={candidate.cross_encoder_rerank_score or 0.0:.6f} | "
            f"cross_encoder_norm="
            f"{candidate.cross_encoder_normalized_score or 0.0:.6f} | "
            f"final={candidate.final_score:.6f} | "
            f"votes={candidate.vote_count}"
        )
        print(
            f"{metadata.get('doc_code', '')} | "
            f"{metadata.get('doc_type', '')} | "
            f"{metadata.get('article', '')}"
        )
        print(candidate.text[:500].replace("\n", " "))


def main() -> None:
    query="Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?"
    latencies = {}

    print("Loading models...")
    load_started = time.perf_counter()
    llm = GroqLLMClient()
    dense_model = load_dense_model()
    colbert_model = load_colbert_model()
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    model_load_latency = time.perf_counter() - load_started
    print(f"[INIT] Load models: {model_load_latency:.3f}s")

    query_started = time.perf_counter()
    started = time.perf_counter()
    understanding = LegalUnderstandingAgent(llm).run(query)
    log_latency("Legal understanding", started, latencies)

    started = time.perf_counter()
    plan = QueryPlannerAgent(llm).run(query, understanding)
    log_latency("Query planning", started, latencies)
    print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))

    qdrant_filter = build_qdrant_filter(plan.filters)
    started = time.perf_counter()
    hybrid_results = run_hybrid(
        plan.queries,
        plan.retrieval,
        dense_model,
        qdrant_filter,
    )

    if not any(hybrid_results) and (plan.filters.domains or plan.filters.sectors):
        print("Không có kết quả taxonomy exact-match, bỏ domain/sector.")
        hybrid_results = run_hybrid(
            plan.queries,
            plan.retrieval,
            dense_model,
            build_qdrant_filter(plan.filters, include_taxonomy=False),
        )

    if not any(hybrid_results):
        print("Không có kết quả với filter định danh, chỉ giữ is_current.")
        hybrid_results = run_hybrid(
            plan.queries,
            plan.retrieval,
            dense_model,
            build_qdrant_filter(
                RetrievalFilter(is_current=plan.filters.is_current)
            ),
        )
    log_latency("Hybrid retrieval", started, latencies)

    exact_results = []
    if plan.retrieval.use_exact:
        print("Exact retrieval...")
        started = time.perf_counter()
        exact_results = exact_search(
            query,
            doc_codes=plan.filters.doc_codes,
            top_k=plan.retrieval.top_k_exact,
        )
        log_latency("Exact retrieval", started, latencies)

    summary_results = []
    if plan.retrieval.use_summary:
        print("Summary retrieval...")
        started = time.perf_counter()
        summary_results = summary_search(
            query,
            top_k=plan.retrieval.top_k_summary,
            current_only=plan.filters.is_current is True,
        )
        log_latency("Summary retrieval", started, latencies)

    started = time.perf_counter()
    query_weights = {
        "original": 1.0,
        "legal_rewrite": 0.8,
        "keyword": 0.6,
    }
    initial_result_sets = list(hybrid_results)
    initial_weights = [
        query_weights.get(search_query.query_type, 0.7)
        for search_query in plan.queries
    ]
    if exact_results:
        initial_result_sets.append(exact_results)
        initial_weights.append(1.2)
    if summary_results:
        initial_result_sets.append(summary_results)
        initial_weights.append(0.5)

    initial_candidates = rrf_fusion(
        initial_result_sets,
        top_k=100,
        weights=initial_weights,
    )
    log_latency("Initial fusion", started, latencies)
    seeds = initial_candidates[:10]

    graph_results = []
    if plan.retrieval.use_graph:
        print("Graph expansion...")
        started = time.perf_counter()
        graph_results = graph_search(
            query,
            seeds,
            top_k=min(plan.retrieval.top_k_graph, 20),
            current_only=plan.filters.is_current is True,
        )
        log_latency("Graph expansion", started, latencies)

    context_results = []
    if plan.retrieval.use_context:
        print("Context expansion...")
        started = time.perf_counter()
        context_results = expand_context(
            [*seeds, *graph_results[:5]],
            query=query,
            top_k=20,
        )
        log_latency("Context expansion", started, latencies)

    started = time.perf_counter()
    expanded_result_sets = [initial_candidates]
    expanded_weights = [1.0]
    if graph_results:
        expanded_result_sets.append(graph_results)
        expanded_weights.append(0.4)
    if context_results:
        expanded_result_sets.append(context_results)
        expanded_weights.append(0.6)
    expanded_candidates = rrf_fusion(
        expanded_result_sets,
        top_k=100,
        weights=expanded_weights,
    )
    log_latency("Expanded fusion", started, latencies)

    if plan.retrieval.use_colbert:
        print("ColBERT rerank 100 -> 60...")
        started = time.perf_counter()
        colbert_candidates = colbert_rerank(
            query,
            expanded_candidates,
            colbert_model,
            top_k=plan.retrieval.top_k_colbert,
        )
        log_latency("ColBERT rerank", started, latencies)
    else:
        colbert_candidates = expanded_candidates[
            :plan.retrieval.top_k_colbert
        ]

    if plan.retrieval.use_cross_encoder:
        print("Cross-encoder rerank 60 -> 40...")
        started = time.perf_counter()
        reranked_candidates = cross_encoder_rerank(
            query,
            colbert_candidates,
            cross_encoder,
            top_k=plan.retrieval.top_k_cross_encoder,
        )
        log_latency("Cross-encoder rerank", started, latencies)
    else:
        reranked_candidates = colbert_candidates[
            :plan.retrieval.top_k_cross_encoder
        ]
    final_candidates = reranked_candidates[:20]
    print("Final top-k: 20")
    print_results(final_candidates)

    selector = EvidenceSelectorAgent(llm)
    started = time.perf_counter()
    document_candidates = selector.get_selected_evidence(
        final_candidates,
    )[:15]
    selection_result = selector.run(
        query,
        document_candidates
    )
    log_latency("Evidence selection", started, latencies)
    selected_ids = {
        item.unit_id for item in selection_result.selected
    }
    selected_evidence = [
        candidate
        for candidate in document_candidates
        if candidate.unit_id in selected_ids
    ]

    print("================ SELECTED EVIDENCE ==================")
    for evidence in selected_evidence:
        print(
            f"{evidence.unit_id} | "
            f"{evidence.metadata.get('doc_code', '')} | "
            f"{evidence.metadata.get('article', '')} | "
            f"score={evidence.final_score:.6f}"
        )

    started = time.perf_counter()
    sufficiency_report = SufficiencyCheckerAgent(llm).run(
        question=query,
        understanding=understanding,
        selection=selection_result,
        evidence=selected_evidence,
    )
    log_latency("Sufficiency check", started, latencies)
    print("================ SUFFICIENCY ==================")
    print(
        json.dumps(
            sufficiency_report.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )

    started = time.perf_counter()
    answer_draft = ReasonerAgent(llm).run(
        question=query,
        understanding=understanding,
        selection=selection_result,
        evidence=selected_evidence,
        sufficiency=sufficiency_report,
    )
    log_latency("Reasoning", started, latencies)
    print("================ ANSWER DRAFT ==================")
    print(answer_draft.answer)

    verifier = VerificationAgent(llm)
    started = time.perf_counter()
    verification_report = verifier.run(
        question=query,
        answer=answer_draft,
        evidence=selected_evidence,
    )
    log_latency("Verification", started, latencies)
    print("================ VERIFICATION ==================")
    print(
        json.dumps(
            verification_report.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )

    if not verification_report.passed:
        started = time.perf_counter()
        answer_draft = ReasonerAgent(llm).run(
            question=query,
            understanding=understanding,
            selection=selection_result,
            evidence=selected_evidence,
            sufficiency=sufficiency_report,
            revision_instruction=verification_report.revision_instruction,
        )
        log_latency("Answer revision", started, latencies)

        started = time.perf_counter()
        verification_report = verifier.run(
            question=query,
            answer=answer_draft,
            evidence=selected_evidence,
        )
        log_latency("Final verification", started, latencies)
        print("================ REVISED ANSWER ==================")
        print(answer_draft.answer)
        print("================ FINAL VERIFICATION ==================")
        print(
            json.dumps(
                verification_report.model_dump(),
                ensure_ascii=False,
                indent=2,
            )
        )

    submission = SubmissionFormatterAgent().run(
        question_id=1,
        question=query,
        answer=answer_draft,
        evidence=selected_evidence,
        verification=verification_report,
    )
    validate_submission_item(submission, selected_evidence)
    output_path = write_results([submission])
    print("================ SUBMISSION ==================")
    print(
        json.dumps(
            submission.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Đã ghi kết quả: {output_path.resolve()}")

    total_query_latency = time.perf_counter() - query_started
    latencies["Total query"] = total_query_latency
    print("================ LATENCY SUMMARY ==================")
    for name, elapsed in latencies.items():
        print(f"{name}: {elapsed:.3f}s")


if __name__ == "__main__":
    main()
