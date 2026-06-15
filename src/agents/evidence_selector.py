import json
from typing import Any

from src.schema.agent_schemas import (
    Evidence,
    EvidenceSelectionResult,
    SelectedEvidence,
)


class EvidenceSelectorAgent:
    def __init__(self, llm: Any):
        self.llm = llm

    def run(
        self,
        question: str,
        candidates: list[Evidence],
        max_candidates: int = 15,
        max_selected: int = 5,
        score_gap: float = 0.1,
    ) -> EvidenceSelectionResult:
        if not candidates:
            return EvidenceSelectionResult(selected=[])

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
                )[:500],
            }
            for candidate in candidates
        ]

        prompt = f"""
Bạn là Evidence Selector cho hệ thống Legal RAG Việt Nam.

Chọn tối đa {max_selected} evidence cần thiết để trả lời câu hỏi.

Quy tắc:
- Chỉ chọn unit_id có trong danh sách.
- Ưu tiên evidence có final_score cao.
- Nếu điểm chênh lệch nhỏ, so sánh nội dung và chọn evidence trả lời
  trực tiếp hơn.
- main: trực tiếp chứa căn cứ trả lời.
- supporting: bổ sung điều kiện, ngoại lệ hoặc giải thích.
- background: chỉ cung cấp bối cảnh cần thiết.
- Không chọn nhiều evidence trùng nội dung.
- Không tự trả lời câu hỏi.

Chỉ trả về JSON:
{{
  "selected": [
    {{
      "unit_id": "...",
      "role": "main | supporting | background",
      "reason": "...",
      "supported_claims": ["..."]
    }}
  ],
  "rejected": [
    {{
      "unit_id": "...",
      "reason": "..."
    }}
  ]
}}
"""
        data = self.llm.call_llm_json(
            query=json.dumps(
                {
                    "question": question,
                    "candidates": evidence_data,
                },
                ensure_ascii=False,
            ),
            system_prompt=prompt,
            max_new_tokens=2048,
            temperature=0.0,
        )

        result = EvidenceSelectionResult(**data)
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
