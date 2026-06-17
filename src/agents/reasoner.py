import json
from typing import Any

from src.schema.agent_schemas import (
    AnswerDraft,
    Evidence,
    EvidenceSelectionResult,
    LegalUnderstanding,
    SufficiencyReport,
)


REASONER_MAX_TOKENS = 400


class ReasonerAgent:
    def __init__(self, llm: Any):
        self.llm = llm

    def run(
        self,
        question: str,
        understanding: LegalUnderstanding,
        selection: EvidenceSelectionResult,
        evidence: list[Evidence],
        sufficiency: SufficiencyReport,
        revision_instruction: str | None = None,
    ) -> AnswerDraft:
        if not evidence or not sufficiency.is_sufficient:
            return AnswerDraft(
                answer="Chưa có đủ căn cứ pháp lý liên quan để trả lời câu hỏi."
            )

        selected_by_id = {item.unit_id: item for item in selection.selected}
        evidence_data = []
        for item in evidence:
            selected = selected_by_id.get(item.unit_id)
            evidence_data.append(
                {
                    "unit_id": item.unit_id,
                    "role": selected.role if selected else "supporting",
                    "selection_reason": selected.reason if selected else None,
                    "supported_claims": (
                        selected.supported_claims if selected else []
                    ),
                    "doc_code": item.doc_code or item.metadata.get("doc_code"),
                    "article": item.article or item.metadata.get("article"),
                    "article_title": (
                        item.article_title
                        or item.metadata.get("article_title")
                    ),
                    "status": item.metadata.get("status"),
                    "text": (
                        item.metadata.get("content_text")
                        or item.text
                    ),
                }
            )

        prompt = """
Bạn là Reasoner Agent của hệ thống hỏi đáp pháp luật Việt Nam.

Nhiệm vụ:
Đọc các evidence đã được chọn, lấy ra những thông tin phù hợp trực tiếp với câu
hỏi và tổng hợp thành câu trả lời chính xác, súc tích.

Nguyên tắc:
- Chỉ sử dụng nội dung có trong evidence, không dùng kiến thức bên ngoài.
- Xem toàn bộ evidence là một tập thông tin chung; một câu trả lời có thể tổng hợp
  nội dung từ nhiều điều và nhiều văn bản khác nhau.
- Chỉ sử dụng các ý trực tiếp trả lời hoặc làm rõ câu hỏi.
- Ưu tiên điều kiện, đối tượng, ngoại lệ và thủ tục cần thiết; bỏ thông tin nền,
  mô tả chung hoặc nội dung không giúp trả lời câu hỏi.
- Evidence có role=supporting hoặc background chỉ được dùng khi bổ sung thông
  tin cần thiết cho câu trả lời.
- Không lặp lại các ý trùng nhau. Chỉ bỏ nội dung hoàn toàn không liên quan đến
  câu hỏi hoặc không làm rõ câu trả lời.
- Không suy diễn, khái quát hoặc tạo điều kiện mới ngoài nội dung được cung cấp.
- Không cần nêu số điều, mã văn bản hoặc chép nguyên văn điều luật.
- Trả lời trực tiếp, không nhắc lại câu hỏi và không viết lời dẫn chung dài dòng.
- Nếu có nhiều ý, trình bày bằng các gạch đầu dòng rõ ràng.
- Nếu evidence chỉ đủ trả lời một phần, trả lời phần có căn cứ và nêu ngắn gọn
  phần chưa có đủ thông tin.
- Nếu có revision_instruction, sửa câu trả lời theo yêu cầu nhưng vẫn phải giữ
  lại mọi nội dung đúng và có căn cứ.
- Không nhắc đến evidence, retrieval, agent hoặc hệ thống trong câu trả lời.
- Trả lời ngắn gọn, ưu tiên các điều kiện trực tiếp liên quan đến câu hỏi.
- Tối đa 5 ý chính; mỗi ý từ 1 đến 2 câu ngắn.
- Không diễn giải dài, không lặp lại cùng một nội dung dưới cách viết khác.
- Tổng câu trả lời nên dưới 250 từ, đúng nội dung nhưng không dài dòng.

Trước khi trả lời, tự kiểm tra:
1. Đã sử dụng toàn bộ các ý liên quan trong evidence chưa?
2. Có bỏ sót điều kiện, ngoại lệ, ưu tiên hoặc thủ tục nào không?
3. Có nội dung nào không được evidence hỗ trợ không?

Chỉ trả về JSON:
{
  "answer": "..."
}
"""
        data = self.llm.call_llm_json(
            query=json.dumps(
                {
                    "question": question,
                    "understanding": understanding.model_dump(),
                    "sufficiency_reason": sufficiency.reason,
                    "revision_instruction": revision_instruction,
                    "evidence": evidence_data,
                },
                ensure_ascii=False,
            ),
            system_prompt=prompt,
            max_new_tokens=REASONER_MAX_TOKENS,
            temperature=0.1,
        )

        answer = str(data.get("answer", "")).strip()
        if not answer:
            raise ValueError("Reasoner không trả về nội dung câu trả lời.")
        return AnswerDraft(answer=answer)
