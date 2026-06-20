import json
from typing import Any

from src.schema.agent_schemas import (
    AnswerDraft,
    Evidence,
    LegalUnderstanding,
)


REASONER_MAX_TOKENS = 512
REASONER_EVIDENCE_CHARS = 768


class ReasonerAgent:
    def __init__(self, llm: Any):
        self.llm = llm

    def run(
        self,
        question: str,
        understanding: LegalUnderstanding,
        evidence: list[Evidence],
    ) -> AnswerDraft:
        if not evidence:
            return AnswerDraft(
                answer="Chưa có đủ căn cứ pháp lý liên quan để trả lời câu hỏi."
            )

        evidence_data = []
        for item in evidence:
            evidence_data.append(
                {
                    "unit_id": item.unit_id,
                    "final_score": item.final_score,
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
                    )[:REASONER_EVIDENCE_CHARS],
                }
            )

        prompt = """
Bạn là Reasoner Agent của hệ thống hỏi đáp pháp luật Việt Nam.

Nhiệm vụ:
Đọc các evidence sau rerank, lấy ra những thông tin phù hợp trực tiếp với câu
hỏi và tổng hợp thành câu trả lời chính xác, súc tích.

Nguyên tắc:
- Chỉ sử dụng nội dung có trong evidence, không dùng kiến thức bên ngoài.
- Xem toàn bộ evidence là một tập thông tin chung; một câu trả lời có thể tổng hợp
  nội dung từ nhiều điều và nhiều văn bản khác nhau.
- Chỉ sử dụng các ý trực tiếp trả lời hoặc làm rõ câu hỏi.
- Phải bao phủ đầy đủ mọi ý trong evidence có liên quan trực tiếp đến câu hỏi;
  không được bỏ sót điều kiện, ngoại lệ, đối tượng, mức hỗ trợ, thủ tục hoặc
  biện pháp quan trọng.
- Được gộp các ý tương đồng để viết ngắn, nhưng không được làm mất nội dung.
- Ưu tiên điều kiện, đối tượng, ngoại lệ và thủ tục cần thiết; bỏ thông tin nền,
  mô tả chung hoặc nội dung không giúp trả lời câu hỏi.
- Không lặp lại các ý trùng nhau. Chỉ bỏ nội dung hoàn toàn không liên quan đến
  câu hỏi hoặc không làm rõ câu trả lời.
- Không suy diễn, khái quát hoặc tạo điều kiện mới ngoài nội dung được cung cấp.
- Không cần nêu số điều, mã văn bản hoặc chép nguyên văn điều luật.
- Trả lời trực tiếp, không nhắc lại câu hỏi và không viết lời dẫn chung dài dòng.
- Nếu có nhiều ý, trình bày tối đa 4 gạch đầu dòng; mỗi ý chỉ một câu.
- Nếu evidence chỉ đủ trả lời một phần, trả lời phần có căn cứ và nêu ngắn gọn
  phần chưa có đủ thông tin.
- Không nhắc đến evidence, retrieval, agent hoặc hệ thống trong câu trả lời.
- Trả lời ngắn gọn, ưu tiên các điều kiện trực tiếp liên quan đến câu hỏi.
- Không diễn giải dài, không lặp lại cùng một nội dung dưới cách viết khác.
- Bắt buộc trả lời tối đa 120 từ. Câu hỏi đơn giản chỉ trả lời từ 1 đến 3 câu.
- Nếu không thể vừa đủ ý vừa giữ 120 từ, ưu tiên trả lời đủ ý và viết cô đọng nhất.
- Không viết câu mở đầu, lời dẫn, nhận xét chung hoặc đoạn kết luận lặp lại nội dung.
- Không được nhắc tới điều, luật, quy định khi trả lời câu hỏi
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
                    "evidence": evidence_data,
                },
                ensure_ascii=False,
            ),
            system_prompt=prompt,
            retry_stage="reasoning",
            max_new_tokens=REASONER_MAX_TOKENS,
            temperature=0.1,
        )

        answer = str(data.get("answer", "")).strip()
        if not answer:
            raise ValueError("Reasoner không trả về nội dung câu trả lời.")
        return AnswerDraft(answer=answer)
