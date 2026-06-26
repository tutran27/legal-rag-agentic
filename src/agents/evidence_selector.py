import json
from typing import Any

from src.schema.agent_schemas import (
    Evidence,
    EvidenceSelectionResult,
    SelectedEvidence,
)


SELECTOR_MAX_TOKENS = 896
SELECTOR_MAX_CANDIDATES = 8
SELECTOR_MAX_SELECTED = 4
SELECTOR_MIN_SELECTED = 2
SELECTOR_TEXT_CHARS = 450


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
- Đầu vào chỉ có {max_candidates} candidates đã rerank.
- Chọn từ 2 đến {max_selected} evidence phù hợp nhất để đưa sang bước reasoning.
- Không nhất thiết chọn đủ {max_selected}; chỉ chọn 4 khi cả 4 đều khớp ngữ cảnh và đều hữu ích cho việc trả lời.
- Nếu evidence nhìn chung khớp câu hỏi thì có thể chọn, không cần quá khắt khe.
- Ưu tiên evidence trả lời trực tiếp câu hỏi, có final_score cao và ít trùng ý.

Quy tắc:
- Chỉ chọn unit_id có trong danh sách candidates.
- Chỉ cần evidence liên quan và có thể dùng để trả lời câu hỏi thì nên giữ lại.
- Ưu tiên 2 hoặc 3 evidence nếu đã đủ để trả lời rõ, nhưng không cần loại bỏ quá mạnh tay, lấy toàn bộ nếu tất cả đều cung cấp thông tin.
- main: chứa căn cứ trực tiếp để trả lời.
- supporting: bổ sung điều kiện, ngoại lệ, thủ tục hoặc hậu quả quan trọng.
- background: chỉ giữ khi thật sự cần để hiểu câu trả lời.
- Được phép chọn các evidence có nội dung gần nhau nếu chúng đều liên quan đến câu hỏi.
- Chỉ cần evidence chứa nội dung liên quan câu hỏi thì có thể cho pass.
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

        fallback_items = self.get_selected_evidence(ranked_candidates)
        min_selected = min(SELECTOR_MIN_SELECTED, len(fallback_items))

        if not selected:
            selected = [
                SelectedEvidence(
                    unit_id=item.unit_id,
                    role="main" if index == 0 else "supporting",
                    reason="Fallback theo final_score.",
                )
                for index, item in enumerate(
                    fallback_items[:min_selected]
                )
            ]
        elif len(selected) < min_selected:
            for item in fallback_items:
                if item.unit_id in seen:
                    continue
                selected.append(
                    SelectedEvidence(
                        unit_id=item.unit_id,
                        role="supporting",
                        reason="Bo sung de dam bao du y chinh.",
                    )
                )
                seen.add(item.unit_id)
                if len(selected) >= min_selected:
                    break

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
