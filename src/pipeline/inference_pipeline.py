import json
import time
from typing import Any

import torch
from qdrant_client import QdrantClient, models
from sentence_transformers import CrossEncoder

from src.agents.evidence_selector import EvidenceSelectorAgent
from src.agents.formatter import SubmissionFormatterAgent
from src.agents.query_planner import QueryPlannerAgent
from src.agents.reasoner import ReasonerAgent
from src.agents.verifier import VerificationAgent
from src.common.config import settings
from src.common.embedding import (
    embed_dense,
    load_colbert_model,
    load_dense_model,
)
from src.generation.endpoint import create_llm_client
from src.retrieval.colbert_reranker import colbert_rerank
from src.retrieval.context_expander import expand_context
from src.retrieval.cross_encoder_rerank import cross_encoder_rerank
from src.retrieval.exact_retriever import exact_search
from src.retrieval.fusion import rrf_fusion
from src.retrieval.graph_retriever import graph_search, load_graph
from src.retrieval.hybrid_retriever import hybrid_search_batch
from src.retrieval.summary_retriever import summary_search
from src.schema.agent_schemas import (
    Evidence,
    InferenceResult,
    RetrievalFilter,
)
from src.submission.validate_results import validate_submission_item


CROSS_ENCODER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
GRAPH_TERMS = {
    "sửa đổi",
    "bổ sung",
    "thay thế",
    "hướng dẫn",
    "dẫn chiếu",
    "liên quan",
    "thi hành",
    "văn bản cha",
}


def build_qdrant_filter(
    filters: RetrievalFilter,
    include_taxonomy: bool = True,
) -> models.Filter | None:
    conditions = []
    fields = (
        ("doc_code", filters.doc_codes),
        ("doc_type", filters.doc_types),
        ("domain", filters.domains if include_taxonomy else []),
        ("sector", filters.sectors if include_taxonomy else []),
    )
    for key, values in fields:
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


class InferencePipeline:
    def __init__(
        self,
        llm: Any | None = None,
        dense_model: Any | None = None,
        colbert_model: Any | None = None,
        cross_encoder: Any | None = None,
        qdrant_client: QdrantClient | None = None,
        preload_graph_index: bool = settings.preload_graph,
        verbose: bool = True,
    ) -> None:
        self.verbose = verbose
        started = time.perf_counter()
        self.llm = llm or create_llm_client()
        self.dense_model = dense_model or load_dense_model()
        self.colbert_model = colbert_model or load_colbert_model()
        self.cross_encoder = cross_encoder or self._load_cross_encoder()
        self.qdrant_client = qdrant_client or QdrantClient(
            url=settings.qdrant_url,
            prefer_grpc=True,
            timeout=settings.qdrant_timeout,
        )
        if preload_graph_index:
            load_graph(settings.graph_path)
        self.model_load_latency = time.perf_counter() - started
        self._print(f"[INIT] Load models: {self.model_load_latency:.3f}s")

    @staticmethod
    def _load_cross_encoder() -> CrossEncoder:
        use_cuda = torch.cuda.is_available()
        return CrossEncoder(
            CROSS_ENCODER_MODEL,
            device="cuda" if use_cuda else "cpu",
            model_kwargs={"torch_dtype": torch.float16} if use_cuda else None,
        )

    def _print(self, message: str = "") -> None:
        if self.verbose:
            print(message)

    def _log_latency(
        self,
        name: str,
        started: float,
        latencies: dict[str, float],
    ) -> None:
        elapsed = time.perf_counter() - started
        latencies[name] = elapsed
        self._print(f"[LATENCY] {name}: {elapsed:.3f}s")

    def close(self) -> None:
        self.qdrant_client.close()

    def _run_hybrid(
        self,
        queries,
        retrieval_plan,
        qdrant_filter,
        dense_vectors,
    ):
        for index, search_query in enumerate(queries, start=1):
            self._print(
                f"Hybrid {index}/{len(queries)}: {search_query.text}"
            )
        return hybrid_search_batch(
            [search_query.text for search_query in queries],
            dense_st=self.dense_model,
            flt=qdrant_filter,
            top_n=min(
                settings.retrieval_top_k,
                max(
                    retrieval_plan.top_k_bm25,
                    retrieval_plan.top_k_dense,
                    retrieval_plan.top_k_sparse,
                ),
            ),
            use_dense=retrieval_plan.use_dense,
            use_sparse=(
                retrieval_plan.use_bm25 or retrieval_plan.use_sparse
            ),
            client=self.qdrant_client,
            dense_vectors=dense_vectors,
        )

    def _retrieve(self, question, plan, latencies):
        query_texts = [query.text for query in plan.queries]
        dense_vectors = (
            embed_dense(query_texts, self.dense_model, is_query=True)
            if plan.retrieval.use_dense
            else None
        )
        started = time.perf_counter()
        hybrid_results = self._run_hybrid(
            plan.queries,
            plan.retrieval,
            build_qdrant_filter(plan.filters),
            dense_vectors,
        )
        if not any(hybrid_results) and (
            plan.filters.domains or plan.filters.sectors
        ):
            self._print(
                "Không có kết quả taxonomy exact-match, bỏ domain/sector."
            )
            hybrid_results = self._run_hybrid(
                plan.queries,
                plan.retrieval,
                build_qdrant_filter(
                    plan.filters,
                    include_taxonomy=False,
                ),
                dense_vectors,
            )
        if not any(hybrid_results):
            self._print(
                "Không có kết quả với filter định danh, chỉ giữ is_current."
            )
            hybrid_results = self._run_hybrid(
                plan.queries,
                plan.retrieval,
                build_qdrant_filter(
                    RetrievalFilter(is_current=plan.filters.is_current)
                ),
                dense_vectors,
            )
        self._log_latency("Hybrid retrieval", started, latencies)

        exact_results = []
        if plan.retrieval.use_exact:
            started = time.perf_counter()
            exact_results = exact_search(
                question,
                doc_codes=plan.filters.doc_codes,
                top_k=min(plan.retrieval.top_k_exact, 20),
                current_only=plan.filters.is_current is True,
                client=self.qdrant_client,
            )
            self._log_latency("Exact retrieval", started, latencies)

        summary_results = []
        if plan.retrieval.use_summary:
            started = time.perf_counter()
            summary_results = summary_search(
                question,
                top_k=plan.retrieval.top_k_summary,
                current_only=plan.filters.is_current is True,
            )
            self._log_latency("Summary retrieval", started, latencies)

        result_sets = list(hybrid_results)
        query_weights = {
            "original": 1.0,
            "legal_rewrite": 0.8,
            "keyword": 0.6,
        }
        weights = [
            query_weights.get(search_query.query_type, 0.7)
            for search_query in plan.queries
        ]
        if exact_results:
            result_sets.append(exact_results)
            weights.append(1.2)
        if summary_results:
            result_sets.append(summary_results)
            weights.append(0.5)

        started = time.perf_counter()
        candidates = rrf_fusion(
            result_sets,
            top_k=settings.initial_fusion_top_k,
            weights=weights,
        )
        self._log_latency("Initial fusion", started, latencies)
        return candidates

    def _retrieve_many(self, planned_items):
        query_texts = [
            query.text
            for _, plan, _ in planned_items
            for query in plan.queries
        ]
        filters = [
            build_qdrant_filter(plan.filters)
            for _, plan, _ in planned_items
            for _ in plan.queries
        ]
        use_dense = all(
            plan.retrieval.use_dense for _, plan, _ in planned_items
        )
        use_sparse = all(
            plan.retrieval.use_bm25 or plan.retrieval.use_sparse
            for _, plan, _ in planned_items
        )
        if not use_dense or not use_sparse:
            return [
                self._retrieve(question, plan, latencies)
                for question, plan, latencies in planned_items
            ]

        started = time.perf_counter()
        dense_vectors = embed_dense(
            query_texts,
            self.dense_model,
            is_query=True,
        )
        flat_results = hybrid_search_batch(
            query_texts,
            dense_st=self.dense_model,
            top_n=settings.retrieval_top_k,
            use_dense=True,
            use_sparse=True,
            client=self.qdrant_client,
            dense_vectors=dense_vectors,
            filters=filters,
        )
        hybrid_elapsed = time.perf_counter() - started

        results = []
        offset = 0
        for question, plan, latencies in planned_items:
            size = len(plan.queries)
            hybrid_results = flat_results[offset : offset + size]
            vector_slice = dense_vectors[offset : offset + size]
            offset += size

            if not any(hybrid_results) and (
                plan.filters.domains or plan.filters.sectors
            ):
                hybrid_results = self._run_hybrid(
                    plan.queries,
                    plan.retrieval,
                    build_qdrant_filter(
                        plan.filters,
                        include_taxonomy=False,
                    ),
                    vector_slice,
                )
            if not any(hybrid_results):
                hybrid_results = self._run_hybrid(
                    plan.queries,
                    plan.retrieval,
                    build_qdrant_filter(
                        RetrievalFilter(
                            is_current=plan.filters.is_current
                        )
                    ),
                    vector_slice,
                )
            latencies["Hybrid retrieval"] = (
                hybrid_elapsed / len(planned_items)
            )

            exact_results = []
            if plan.retrieval.use_exact:
                exact_started = time.perf_counter()
                exact_results = exact_search(
                    question,
                    doc_codes=plan.filters.doc_codes,
                    top_k=min(plan.retrieval.top_k_exact, 20),
                    current_only=plan.filters.is_current is True,
                    client=self.qdrant_client,
                )
                latencies["Exact retrieval"] = (
                    time.perf_counter() - exact_started
                )

            result_sets = list(hybrid_results)
            query_weights = {
                "original": 1.0,
                "legal_rewrite": 0.8,
                "keyword": 0.6,
            }
            weights = [
                query_weights.get(search_query.query_type, 0.7)
                for search_query in plan.queries
            ]
            if exact_results:
                result_sets.append(exact_results)
                weights.append(1.2)
            if plan.retrieval.use_summary:
                summary_started = time.perf_counter()
                summary_results = summary_search(
                    question,
                    top_k=plan.retrieval.top_k_summary,
                    current_only=plan.filters.is_current is True,
                )
                latencies["Summary retrieval"] = (
                    time.perf_counter() - summary_started
                )
                if summary_results:
                    result_sets.append(summary_results)
                    weights.append(0.5)
            fusion_started = time.perf_counter()
            results.append(
                rrf_fusion(
                    result_sets,
                    top_k=settings.initial_fusion_top_k,
                    weights=weights,
                )
            )
            latencies["Initial fusion"] = (
                time.perf_counter() - fusion_started
            )
        return results

    @staticmethod
    def _needs_graph(question: str, candidates: list[Evidence]) -> bool:
        normalized = question.lower()
        if any(term in normalized for term in GRAPH_TERMS):
            return True
        return bool(candidates and candidates[0].vote_count < 2)

    @staticmethod
    def _needs_context(candidates: list[Evidence]) -> bool:
        return any(
            int(candidate.metadata.get("part_count") or 1) > 1
            and candidate.metadata.get("part_index") is not None
            for candidate in candidates
        )

    def _expand(self, question, plan, candidates, latencies):
        seeds = candidates[:settings.graph_seed_top_k]
        graph_results = []
        if (
            plan.retrieval.use_graph
            and self._needs_graph(question, candidates)
        ):
            started = time.perf_counter()
            graph_results = graph_search(
                question,
                seeds,
                top_k=min(
                    plan.retrieval.top_k_graph,
                    settings.graph_top_k,
                ),
                current_only=plan.filters.is_current is True,
                client=self.qdrant_client,
                graph_path=settings.graph_path,
            )
            self._log_latency("Graph expansion", started, latencies)

        context_results = []
        context_seeds = [*seeds, *graph_results[:3]]
        if (
            plan.retrieval.use_context
            and self._needs_context(context_seeds)
        ):
            started = time.perf_counter()
            context_results = expand_context(
                context_seeds,
                query=question,
                top_k=settings.context_top_k,
                client=self.qdrant_client,
            )
            self._log_latency("Context expansion", started, latencies)

        result_sets = [candidates]
        weights = [1.0]
        if graph_results:
            result_sets.append(graph_results)
            weights.append(0.4)
        if context_results:
            result_sets.append(context_results)
            weights.append(0.6)

        started = time.perf_counter()
        expanded = rrf_fusion(
            result_sets,
            top_k=settings.initial_fusion_top_k,
            weights=weights,
        )
        self._log_latency("Expanded fusion", started, latencies)
        return expanded

    def _rerank(self, question, retrieval_plan, candidates, latencies):
        if retrieval_plan.use_colbert:
            started = time.perf_counter()
            candidates = colbert_rerank(
                question,
                candidates,
                self.colbert_model,
                top_k=min(
                    retrieval_plan.top_k_colbert,
                    settings.colbert_top_k,
                ),
            )
            self._log_latency("ColBERT rerank", started, latencies)
        else:
            candidates = candidates[:settings.colbert_top_k]

        if retrieval_plan.use_cross_encoder:
            started = time.perf_counter()
            candidates = cross_encoder_rerank(
                question,
                candidates,
                self.cross_encoder,
                top_k=min(
                    retrieval_plan.top_k_cross_encoder,
                    settings.cross_encoder_top_k,
                ),
            )
            self._log_latency("Cross-encoder rerank", started, latencies)
        else:
            candidates = candidates[:settings.cross_encoder_top_k]
        return candidates[:settings.final_top_k]

    def _print_results(self, candidates: list[Evidence]) -> None:
        self._print(f"\n=== KẾT QUẢ CUỐI ({len(candidates)}) ===")
        for rank, candidate in enumerate(candidates, start=1):
            metadata = candidate.metadata
            self._print(
                f"\n[{rank}] chunk={candidate.chunk_id} "
                f"source={candidate.source}"
            )
            self._print(
                f"scores: fusion={candidate.score:.6f} | "
                f"colbert={candidate.colbert_rerank_score or 0.0:.6f} | "
                f"colbert_norm="
                f"{candidate.colbert_normalized_score or 0.0:.6f} | "
                f"cross_encoder="
                f"{candidate.cross_encoder_rerank_score or 0.0:.6f} | "
                f"cross_encoder_norm="
                f"{candidate.cross_encoder_normalized_score or 0.0:.6f} | "
                f"final={candidate.final_score:.6f} | "
                f"votes={candidate.vote_count}"
            )
            self._print(
                f"{metadata.get('doc_code', '')} | "
                f"{metadata.get('doc_type', '')} | "
                f"{metadata.get('article', '')}"
            )
            self._print(candidate.text[:500].replace("\n", " "))

    def _complete(
        self,
        question,
        question_id,
        understanding,
        plan,
        candidates,
        latencies,
        query_started,
    ) -> InferenceResult:
        candidates = self._expand(question, plan, candidates, latencies)
        final_candidates = self._rerank(
            question,
            plan.retrieval,
            candidates,
            latencies,
        )
        self._print_results(final_candidates)

        selector = EvidenceSelectorAgent(self.llm)
        document_candidates = selector.get_selected_evidence(
            final_candidates
        )[:15]
        started = time.perf_counter()
        assessment = selector.run_with_sufficiency(
            question=question,
            understanding=understanding,
            candidates=document_candidates,
        )
        self._log_latency(
            "Evidence selection + sufficiency",
            started,
            latencies,
        )
        selected_ids = {
            item.unit_id for item in assessment.selection.selected
        }
        selected_evidence = [
            candidate
            for candidate in document_candidates
            if candidate.unit_id in selected_ids
        ]

        started = time.perf_counter()
        reasoner = ReasonerAgent(self.llm)
        answer = reasoner.run(
            question=question,
            understanding=understanding,
            selection=assessment.selection,
            evidence=selected_evidence,
            sufficiency=assessment.sufficiency,
        )
        self._log_latency("Reasoning", started, latencies)

        verifier = VerificationAgent(self.llm)
        started = time.perf_counter()
        verification = verifier.run(
            question=question,
            answer=answer,
            evidence=selected_evidence,
        )
        self._log_latency("Verification", started, latencies)

        if not verification.passed:
            started = time.perf_counter()
            answer = reasoner.run(
                question=question,
                understanding=understanding,
                selection=assessment.selection,
                evidence=selected_evidence,
                sufficiency=assessment.sufficiency,
                revision_instruction=verification.revision_instruction,
            )
            self._log_latency("Answer revision", started, latencies)

            started = time.perf_counter()
            verification = verifier.run(
                question=question,
                answer=answer,
                evidence=selected_evidence,
            )
            self._log_latency("Final verification", started, latencies)

        submission = SubmissionFormatterAgent().run(
            question_id=question_id,
            question=question,
            answer=answer,
            evidence=selected_evidence,
            verification=verification,
        )
        validate_submission_item(submission, selected_evidence)
        latencies["Total query"] = time.perf_counter() - query_started
        return InferenceResult(
            submission=submission,
            final_candidates=final_candidates,
            selected_evidence=selected_evidence,
            verification=verification,
            latencies=latencies,
        )

    def run(self, question: str, question_id: int = 1) -> InferenceResult:
        latencies: dict[str, float] = {}
        query_started = time.perf_counter()
        started = time.perf_counter()
        planning = QueryPlannerAgent(self.llm).run_combined(question)
        self._log_latency(
            "Understanding + query planning",
            started,
            latencies,
        )
        self._print(
            json.dumps(planning.plan.model_dump(), ensure_ascii=False, indent=2)
        )
        candidates = self._retrieve(question, planning.plan, latencies)
        return self._complete(
            question,
            question_id,
            planning.understanding,
            planning.plan,
            candidates,
            latencies,
            query_started,
        )

    def run_many(
        self,
        items: list[tuple[int, str]],
    ) -> list[InferenceResult | Exception]:
        planned = []
        outputs: list[InferenceResult | Exception | None] = [
            None
        ] * len(items)
        for index, (question_id, question) in enumerate(items):
            latencies: dict[str, float] = {}
            query_started = time.perf_counter()
            started = time.perf_counter()
            try:
                planning = QueryPlannerAgent(self.llm).run_combined(question)
                latencies["Understanding + query planning"] = (
                    time.perf_counter() - started
                )
                planned.append(
                    (
                        index,
                        question_id,
                        question,
                        planning,
                        latencies,
                        query_started,
                    )
                )
            except Exception as error:
                outputs[index] = error

        if planned:
            candidates_by_item = self._retrieve_many(
                [
                    (question, planning.plan, latencies)
                    for (
                        _,
                        _,
                        question,
                        planning,
                        latencies,
                        _,
                    ) in planned
                ]
            )
            for planned_item, candidates in zip(
                planned,
                candidates_by_item,
            ):
                (
                    index,
                    question_id,
                    question,
                    planning,
                    latencies,
                    query_started,
                ) = planned_item
                try:
                    outputs[index] = self._complete(
                        question,
                        question_id,
                        planning.understanding,
                        planning.plan,
                        candidates,
                        latencies,
                        query_started,
                    )
                except Exception as error:
                    outputs[index] = error
        return [
            output
            if output is not None
            else RuntimeError("Pipeline không trả về kết quả.")
            for output in outputs
        ]
