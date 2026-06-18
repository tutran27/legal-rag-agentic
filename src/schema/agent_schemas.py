from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class TestQuestion(BaseModel):
    id: int
    question: str


class SearchQuery(BaseModel):
    query_type: Literal[
        "original",
        "legal_rewrite",
        "keyword",
        "hyde",
        "graph",
        "fallback",
    ]
    text: str
    reason: str | None = None


class RetrievalFilter(BaseModel):
    doc_codes: list[str] = Field(default_factory=list)
    doc_types: list[str] = Field(default_factory=list)
    domains: list[str] = Field(default_factory=list)
    sectors: list[str] = Field(default_factory=list)
    is_current: bool | None = True


class RetrievalPlan(BaseModel):
    use_exact: bool = True
    use_bm25: bool = True
    use_dense: bool = True
    use_sparse: bool = True
    use_colbert: bool = True
    use_cross_encoder: bool = True
    use_graph: bool = True
    use_context: bool = True
    use_summary: bool = False

    top_k_exact: int = 20
    top_k_bm25: int = 40
    top_k_dense: int = 40
    top_k_sparse: int = 40
    top_k_colbert: int = 20
    top_k_cross_encoder: int = 10
    top_k_graph: int = 10
    top_k_summary: int = 30


class QueryPlan(BaseModel):
    queries: list[SearchQuery]
    filters: RetrievalFilter = Field(default_factory=RetrievalFilter)
    retrieval: RetrievalPlan = Field(default_factory=RetrievalPlan)


class PlanningResult(BaseModel):
    plan: QueryPlan


class Evidence(BaseModel):
    unit_id: str
    chunk_id: str | None = None

    text: str

    doc_code: str | None = None
    doc_title_submission: str | None = None
    article: str | None = None
    article_title: str | None = None

    source: str | None = None
    chunk_type: str | None = None

    score: float = 0.0
    final_score: float = 0.0
    rerank_score: float | None = None
    colbert_rerank_score: float | None = None
    colbert_normalized_score: float | None = None
    cross_encoder_rerank_score: float | None = None
    cross_encoder_normalized_score: float | None = None
    vote_count: int = 0

    metadata: dict[str, Any] = Field(default_factory=dict)


class AnswerDraft(BaseModel):
    answer: str


class SubmissionItem(BaseModel):
    id: int
    question: str
    answer: str
    relevant_docs: list[str]
    relevant_articles: list[str]


class InferenceResult(BaseModel):
    submission: SubmissionItem
    final_candidates: list[Evidence] = Field(default_factory=list)
    selected_evidence: list[Evidence] = Field(default_factory=list)
    latencies: dict[str, float] = Field(default_factory=dict)


class AgentState(BaseModel):
    id: int
    question: str

    search_queries: list[SearchQuery] = Field(default_factory=list)
    retrieval_filters: RetrievalFilter = Field(default_factory=RetrievalFilter)
    retrieval_plan: RetrievalPlan | None = None

    raw_candidates: list[Evidence] = Field(default_factory=list)
    fused_candidates: list[Evidence] = Field(default_factory=list)
    filtered_candidates: list[Evidence] = Field(default_factory=list)
    reranked_candidates: list[Evidence] = Field(default_factory=list)
    expanded_candidates: list[Evidence] = Field(default_factory=list)
    selected_evidence: list[Evidence] = Field(default_factory=list)

    answer: str | None = None
    submission_item: SubmissionItem | None = None

    retry_count: int = 0
    errors: list[str] = Field(default_factory=list)
