import json
from typing import Any

from src.schema.agent_schemas import (
    Evidence,
    EvidenceSelectionResult,
    SelectedEvidence,
)


SELECTOR_MAX_TOKENS = 256
SELECTOR_MAX_CANDIDATES = 6
SELECTOR_MAX_SELECTED = 4
SELECTOR_TEXT_CHARS = 400


class EvidenceSelectorAgent:
    def __init__(self, llm: Any):
        self.llm = llm

    def run(
        self,
        question: str,
        candidates: list[Evidence],
        max_candidates: int = SELECTOR_MAX_CANDIDATES,
        max_selected: int = SELECTOR_MAX_SELECTED,
    ) -> EvidenceSelectionResult:
        if not candidates:
            return EvidenceSelectionResult(selected=[])

        ranked_candidates = candidates[:max_candidates]
        evidence_data = [
            {
                "unit_id": candidate.unit_id,
                "doc_code": candidate.doc_code or candidate.metadata.get("doc_code"),
                "article": candidate.article or candidate.metadata.get("article"),
                "final_score": candidate.final_score,
                "text": (
                    candidate.metadata.get("content_text")
                    or candidate.text
                )[:SELECTOR_TEXT_CHARS],
            }
            for candidate in ranked_candidates
        ]

        prompt = f"""
Bạn là Evidence Selector cho hệ thống Legal RAG Việt Nam.

Nhiệm vụ:
- Chọn tối đa {max_selected} evidence đáng tin nhất để đưa sang bước reasoning.
- Ưu tiên evidence trả lời trực tiếp câu hỏi, có final_score cao và ít trùng ý.

Quy tắc:
- Chỉ chọn unit_id có trong danh sách candidates.
- main: chứa căn cứ trực tiếp để trả lời.
- supporting: bổ sung điều kiện, ngoại lệ, thủ tục hoặc hậu quả quan trọng.
- background: chỉ giữ khi thật sự cần để hiểu câu trả lời.
- Không chọn nhiều evidence trùng nhau về nội dung.
- Không tự trả lời câu hỏi.

Chỉ trả về JSON:
{{
  "selected": [
    {{
      "unit_id": "...",
      "role": "main",
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
            retry_stage="evidence_selection",
            max_new_tokens=SELECTOR_MAX_TOKENS,
            temperature=0.0,
        )
        result = EvidenceSelectionResult(**data)
        valid_ids = {candidate.unit_id for candidate in ranked_candidates}
        selected = []
        seen = set()
        for item in result.selected:
            if item.unit_id not in valid_ids or item.unit_id in seen:
                continue
            seen.add(item.unit_id)
            selected.append(item)

        if not selected:
            selected = [
                SelectedEvidence(
                    unit_id=item.unit_id,
                    role="main" if index == 0 else "supporting",
                    reason="Fallback theo final_score.",
                )
                for index, item in enumerate(
                    self.get_selected_evidence(ranked_candidates)[:max_selected]
                )
            ]

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

