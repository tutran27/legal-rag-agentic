import json
from typing import Any

from src.schema.agent_schemas import (
    AnswerDraft,
    Evidence,
    VerificationReport,
)


VERIFIER_MAX_TOKENS = 128
VERIFIER_EVIDENCE_CHARS = 900


class VerificationAgent:
    def __init__(self, llm: Any):
        self.llm = llm

    def run(
        self,
        question: str,
        answer: AnswerDraft,
        evidence: list[Evidence],
    ) -> VerificationReport:
        if not answer.answer.strip() or not evidence:
            return VerificationReport(
                passed=False,
                unsupported_claims=["Không có câu trả lời hoặc căn cứ để kiểm tra."],
                revision_instruction=(
                    "Chỉ trả lời khi có căn cứ pháp lý phù hợp."
                ),
            )

        evidence_data = [
            {
                "unit_id": item.unit_id,
                "doc_code": item.doc_code or item.metadata.get("doc_code"),
                "article": item.article or item.metadata.get("article"),
                "status": item.metadata.get("status"),
                "text": (
                    item.metadata.get("content_text") or item.text
                )[:VERIFIER_EVIDENCE_CHARS],
            }
            for item in evidence
        ]

        prompt = """
Bạn là Verification Agent cho hệ thống Legal RAG Việt Nam.

Nhiệm vụ duy nhất là kiểm tra câu trả lời có phù hợp với toàn bộ selected evidence hay không.

Quy tắc:
- Trả passed=true nếu nội dung câu trả lời đúng hoặc có ý nghĩa tương đương với
  thông tin trong bất kỳ selected evidence nào.
- Xem toàn bộ selected evidence là một tập thông tin chung. Cho phép tổng hợp và
  ghép các ý từ nhiều evidence khác nhau.
- Cho phép diễn giải, rút gọn, đổi cách dùng từ và trình bày lại miễn không làm
  thay đổi ý nghĩa chính.
- Không yêu cầu câu trả lời phải sử dụng hết evidence hoặc bao phủ mọi chi tiết.
- Không yêu cầu số điều, mã văn bản hoặc trích dẫn.
- missing_citations và extra_citations luôn trả về danh sách rỗng.
- Chỉ trả passed=false khi câu trả lời chứa thông tin rõ ràng trái với selected
  evidence hoặc đưa thêm một kết luận quan trọng không có trong bất kỳ evidence nào.
- Không bắt lỗi vì câu trả lời ngắn, thiếu chi tiết phụ hoặc dùng cách diễn đạt khác.
- Khi false, unsupported_claims chỉ ghi đúng nội dung nằm ngoài hoặc trái evidence.
- revision_instruction chỉ yêu cầu bỏ hoặc sửa đúng nội dung sai đó.

Chỉ trả về JSON:
{
  "passed": false,
  "unsupported_claims": ["..."],
  "missing_citations": ["..."],
  "extra_citations": ["..."],
  "revision_instruction": "..."
}
"""
        data = self.llm.call_llm_json(
            query=json.dumps(
                {
                    "question": question,
                    "answer": answer.answer,
                    "evidence": evidence_data,
                },
                ensure_ascii=False,
            ),
            system_prompt=prompt,
            max_new_tokens=VERIFIER_MAX_TOKENS,
            temperature=0.0,
        )

        report = VerificationReport(**data)
        report = report.model_copy(
            update={
                "missing_citations": [],
                "extra_citations": [],
            }
        )
        has_errors = bool(report.unsupported_claims)
        if not has_errors:
            return report.model_copy(
                update={"passed": True, "revision_instruction": None}
            )

        instruction = report.revision_instruction
        if not instruction:
            instruction = (
                "Bỏ hoặc sửa các kết luận không được nội dung cung cấp hỗ trợ."
            )
        return report.model_copy(
            update={
                "passed": False,
                "revision_instruction": instruction,
            }
        )
