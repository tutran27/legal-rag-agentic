import json
from typing import Any

from src.schema.agent_schemas import (
    Evidence,
    EvidenceSelectionResult,
    LegalUnderstanding,
    SufficiencyReport,
)


SUFFICIENCY_MAX_TOKENS = 650


class SufficiencyCheckerAgent:
    def __init__(self, llm: Any):
        self.llm = llm

    def run(
        self,
        question: str,
        understanding: LegalUnderstanding,
        selection: EvidenceSelectionResult,
        evidence: list[Evidence],
    ) -> SufficiencyReport:
        if not evidence:
            return SufficiencyReport(
                is_sufficient=False,
                reason="Chưa có evidence liên quan đến câu hỏi.",
                missing_evidence=["Căn cứ pháp lý liên quan"],
                next_queries=[question],
            )

        roles = {item.unit_id: item for item in selection.selected}
        evidence_data = [
            {
                "unit_id": item.unit_id,
                "role": roles[item.unit_id].role if item.unit_id in roles else None,
                "supported_claims": (
                    roles[item.unit_id].supported_claims
                    if item.unit_id in roles
                    else []
                ),
                "doc_code": item.metadata.get("doc_code"),
                "article": item.metadata.get("article"),
                "status": item.metadata.get("status"),
                "text": item.text,
            }
            for item in evidence
        ]

        prompt = """
Bạn là Sufficiency Checker cho hệ thống Legal RAG Việt Nam.

Nhiệm vụ chính là kiểm tra evidence có liên quan đến câu hỏi hay không.

Quy tắc:
- Trả is_sufficient=true nếu ít nhất một evidence đề cập đúng nội dung,
  cùng chủ đề hoặc có ý nghĩa tương tự câu hỏi.
- Không yêu cầu evidence phải trả lời đầy đủ mọi điều kiện, ngoại lệ,
  thủ tục hoặc mọi sub-question.
- Chỉ cần evidence có thể dùng làm căn cứ để trả lời đúng hướng là đủ.
- Câu hỏi và văn bản có thể dùng từ khác nhau; đánh giá theo ý nghĩa,
  không yêu cầu khớp từ khóa chính xác.
- Văn bản hết hiệu lực một phần vẫn được xem là evidence liên quan.
- Chỉ trả is_sufficient=false khi toàn bộ evidence không liên quan rõ ràng
  với câu hỏi hoặc không có nội dung nào có thể dùng để trả lời.

Nếu false:
- Nêu ngắn gọn chủ đề đang bị lệch hoặc nội dung cần tìm.
- Tạo tối đa 3 query mới nhắm đúng nội dung câu hỏi.

Chỉ trả về JSON:
{
  "is_sufficient": true,
  "reason": "...",
  "missing_evidence": [],
  "next_queries": []
}
"""
        data = self.llm.call_llm_json(
            query=json.dumps(
                {
                    "question": question,
                    "understanding": understanding.model_dump(),
                    "evidence": evidence_data,
                },
                ensure_ascii=False,
            ),
            system_prompt=prompt,
            max_new_tokens=SUFFICIENCY_MAX_TOKENS,
            temperature=0.0,
        )

        report = SufficiencyReport(**data)
        if report.is_sufficient:
            return report.model_copy(
                update={
                    "missing_evidence": [],
                    "next_queries": [],
                }
            )

        next_queries = [
            query.strip()
            for query in report.next_queries
            if query.strip() and query.strip() != question.strip()
        ]
        if not next_queries:
            next_queries = report.missing_evidence[:3] or [question]

        return report.model_copy(
            update={"next_queries": next_queries[:3]}
        )
