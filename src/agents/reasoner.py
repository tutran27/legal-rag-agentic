import json
from typing import Any

from src.schema.agent_schemas import AnswerDraft, Evidence


REASONER_MAX_TOKENS = 384
REASONER_EVIDENCE_CHARS = 900
FALLBACK_ANSWER = "Chưa có đủ căn cứ pháp lý liên quan để trả lời câu hỏi."


class ReasonerAgent:
    def __init__(self, llm: Any):
        self.llm = llm

    def run(
        self,
        question: str,
        evidence: list[Evidence],
        revision_instruction: str | None = None,
    ) -> AnswerDraft:
        if not evidence:
            return AnswerDraft(answer=FALLBACK_ANSWER)

        data = self.llm.call_llm_json(
            query=json.dumps(
                {
                    "question": question,
                    "revision_instruction": revision_instruction,
                    "evidence": self._build_evidence_payload(evidence),
                },
                ensure_ascii=False,
            ),
            system_prompt=REASONER_PROMPT,
            max_new_tokens=REASONER_MAX_TOKENS,
            temperature=0.1,
        )

        answer = str(data.get("answer", "")).strip()
        if not answer:
            raise ValueError("Reasoner không trả về nội dung câu trả lời.")
        return AnswerDraft(answer=answer)

    @staticmethod
    def _build_evidence_payload(evidence: list[Evidence]) -> list[dict[str, Any]]:
        payload = []
        for index, item in enumerate(evidence):
            payload.append(
                {
                    "unit_id": item.unit_id,
                    "role": "main" if index == 0 else "supporting",
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
        return payload


REASONER_PROMPT = """
Bạn là Reasoner Agent của hệ thống hỏi đáp pháp luật Việt Nam.

Nhiệm vụ:
Đọc các evidence đã được chọn, lấy ra thông tin phù hợp trực tiếp với câu hỏi
và tổng hợp thành câu trả lời chính xác, súc tích.

Nguyên tắc:
- Chỉ sử dụng nội dung có trong evidence, không dùng kiến thức bên ngoài.
- Có thể tổng hợp nội dung từ nhiều điều và nhiều văn bản khác nhau.
- Ưu tiên điều kiện, đối tượng, ngoại lệ và thủ tục cần thiết.
- Bỏ thông tin nền, mô tả chung hoặc nội dung không giúp trả lời câu hỏi.
- Không suy diễn, khái quát hóa hoặc tạo điều kiện mới ngoài evidence.
- Không cần nêu số điều, mã văn bản hoặc chép nguyên văn điều luật.
- Không nhắc đến evidence, retrieval, agent hoặc hệ thống trong câu trả lời.
- Nếu có nhiều ý, trình bày bằng các gạch đầu dòng rõ ràng.
- Nếu evidence chỉ đủ trả lời một phần, trả lời phần có căn cứ và nêu ngắn gọn
  phần chưa có đủ thông tin.
- Nếu có revision_instruction, sửa câu trả lời theo yêu cầu nhưng vẫn chỉ dựa
  trên evidence.
- Tối đa 5 ý chính; mỗi ý từ 1 đến 2 câu ngắn.
- Tổng câu trả lời nên dưới 250 từ.

Chỉ trả về JSON:
{
  "answer": "..."
}
"""
