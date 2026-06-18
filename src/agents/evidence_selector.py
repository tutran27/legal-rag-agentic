import json
from typing import Any

from src.schema.agent_schemas import (
    Evidence,
    EvidenceAssessment,
    EvidenceSelectionResult,
    LegalUnderstanding,
    SelectedEvidence,
    SufficiencyReport,
)


EVIDENCE_ASSESSMENT_MAX_TOKENS = 384
EVIDENCE_TEXT_CHARS = 600


class EvidenceSelectorAgent:
    def __init__(self, llm: Any):
        self.llm = llm

    def run(
        self,
        question: str,
        understanding: LegalUnderstanding,
        candidates: list[Evidence],
        max_candidates: int = 10,
        max_selected: int = 5,
        score_gap: float = 0.1,
    ) -> EvidenceAssessment:
        if not candidates:
            return EvidenceAssessment(
                selection=EvidenceSelectionResult(selected=[]),
                sufficiency=SufficiencyReport(
                    is_sufficient=False,
                    reason="Chưa có evidence liên quan đến câu hỏi.",
                    missing_evidence=["Căn cứ pháp lý liên quan"],
                    next_queries=[question],
                ),
            )

        candidates = candidates[:max_candidates]
        evidence_data = [
            {
                "unit_id": candidate.unit_id,
                "doc_code": candidate.metadata.get("doc_code"),
                "article": candidate.metadata.get("article"),
                "status": candidate.metadata.get("status"),
                "final_score": candidate.final_score,
                "text": (
                    candidate.metadata.get("content_text")
                    or candidate.text
                )[:EVIDENCE_TEXT_CHARS],
            }
            for candidate in candidates
        ]
        prompt = f"""
Bạn là Evidence Selector và Sufficiency Checker cho Legal RAG Việt Nam.

Trong một lần xử lý:
1. Chọn tối đa {max_selected} evidence trực tiếp hoặc cần thiết nhất.
2. Đánh giá evidence đã chọn có liên quan đủ để trả lời đúng hướng hay không.

Quy tắc:
- Chỉ chọn unit_id trong danh sách.
- Ưu tiên final_score cao; nếu điểm gần nhau thì ưu tiên nội dung trực tiếp hơn.
- Không chọn nhiều evidence trùng nội dung.
- is_sufficient=true khi ít nhất một evidence đã chọn có thể làm căn cứ trả lời.
- Chỉ false khi toàn bộ evidence lệch chủ đề hoặc không thể dùng để trả lời.
- Khi false, tạo tối đa 3 query mới.

Chỉ trả JSON:
{{
  "selection": {{
    "selected": [
      {{
        "unit_id": "...",
        "role": "main | supporting | background",
        "reason": "...",
        "supported_claims": ["..."]
      }}
    ],
    "rejected": []
  }},
  "sufficiency": {{
    "is_sufficient": true,
    "reason": "...",
    "missing_evidence": [],
    "next_queries": []
  }}
}}
"""
        data = self.llm.call_llm_json(
            query=json.dumps(
                {
                    "question": question,
                    "understanding": understanding.model_dump(),
                    "candidates": evidence_data,
                },
                ensure_ascii=False,
            ),
            system_prompt=prompt,
            max_new_tokens=EVIDENCE_ASSESSMENT_MAX_TOKENS,
            temperature=0.0,
        )
        assessment = EvidenceAssessment(**data)
        selection = self._normalize_selection(
            assessment.selection,
            candidates,
            max_selected,
            score_gap,
        )
        sufficiency = assessment.sufficiency
        if not selection.selected:
            sufficiency = sufficiency.model_copy(
                update={
                    "is_sufficient": False,
                    "next_queries": sufficiency.next_queries or [question],
                }
            )
        elif sufficiency.is_sufficient:
            sufficiency = sufficiency.model_copy(
                update={"missing_evidence": [], "next_queries": []}
            )
        else:
            next_queries = [
                query.strip()
                for query in sufficiency.next_queries
                if query.strip() and query.strip() != question.strip()
            ]
            sufficiency = sufficiency.model_copy(
                update={
                    "next_queries": (
                        next_queries[:3]
                        or sufficiency.missing_evidence[:3]
                        or [question]
                    )
                }
            )
        return EvidenceAssessment(
            selection=selection,
            sufficiency=sufficiency,
        )

    def _normalize_selection(
        self,
        result: EvidenceSelectionResult,
        candidates: list[Evidence],
        max_selected: int,
        score_gap: float,
    ) -> EvidenceSelectionResult:
        valid_ids = {candidate.unit_id for candidate in candidates}
        seen = set()
        selected = []
        for item in result.selected:
            if item.unit_id not in valid_ids or item.unit_id in seen:
                continue
            seen.add(item.unit_id)
            selected.append(item)

        if len(candidates) > 1:
            top = candidates[0]
            if (
                top.final_score - candidates[1].final_score > score_gap
                and top.unit_id not in seen
            ):
                selected.insert(
                    0,
                    SelectedEvidence(
                        unit_id=top.unit_id,
                        role="main",
                        reason="Evidence có final_score cao vượt trội.",
                    ),
                )

        return result.model_copy(update={"selected": selected[:max_selected]})

    def get_selected_evidence(
        self,
        candidates: list[Evidence],
    ) -> list[Evidence]:
        best_candidates: dict[tuple[str, str], Evidence] = {}
        for candidate in candidates:
            doc_code = (
                candidate.doc_code
                or candidate.metadata.get("doc_code")
                or candidate.metadata.get("doc_id")
            )
            article = candidate.article or candidate.metadata.get("article")
            article_key = (
                (str(doc_code), str(article))
                if doc_code and article
                else (candidate.unit_id, "")
            )
            current = best_candidates.get(article_key)
            if current is None or candidate.final_score > current.final_score:
                best_candidates[article_key] = candidate

        return sorted(
            best_candidates.values(),
            key=lambda candidate: candidate.final_score,
            reverse=True,
        )
