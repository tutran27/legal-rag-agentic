import json
from typing import Any

from src.schema.agent_schemas import (
    AnswerDraft,
    Evidence,
    LegalUnderstanding,
)


REASONER_MAX_TOKENS = 512
REASONER_EVIDENCE_CHARS = 512


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
                answer="Chua co du can cu phap ly lien quan de tra loi cau hoi."
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

* Đọc câu hỏi người dùng và selected evidence/candidates.
* Trả lời đúng trọng tâm câu hỏi bằng thông tin có trong evidence/candidates.
* Chỉ lấy các ý phù hợp để trả lời, không bê nguyên toàn bộ nội dung.

Quy tắc chọn thông tin:

* Ưu tiên evidence/candidates khớp trực tiếp với câu hỏi.
* LƯU Ý: Một số evidence/candidates có thể không nhắc lại đúng từ ngữ trong câu hỏi mà nó là dạng câu trả lời, mình vẫn có thể dùng để suy ra câu trả lời nếu cùng chủ thể, hành vi, điều kiện, quyền, nghĩa vụ, chế tài hoặc hệ quả pháp lý. Cần đặc biệt lưu ý chỗ này tránh bỏ sót các câu trả lời quan trọng
* Nếu không có đoạn khớp trực tiếp, vẫn được dùng evidence/candidates có nội dung pháp lý liên quan và đủ cơ sở để suy ra câu trả lời.
* Có thể chọn đoạn không nhắc lại đúng từ ngữ trong câu hỏi, miễn là cùng chủ thể, hành vi, điều kiện, quyền, nghĩa vụ, chế tài hoặc hệ quả pháp lý.
* Không dùng đoạn chỉ giống từ khóa nhưng sai ngữ cảnh pháp lý.
* Không thêm kiến thức ngoài evidence/candidates.

Quy tắc trả lời:

* Trả lời thẳng vào điều người dùng hỏi.
* Giữ lại các ý quan trọng nếu liên quan trực tiếp: điều kiện, đối tượng, ngoại lệ, thủ tục, mức xử lý, quyền, nghĩa vụ.
* Gộp ý trùng lặp để câu trả lời ngắn hơn.
* Không nhắc đến điều luật, số văn bản, evidence, retrieval, candidates, agent hay hệ thống.
* Không viết câu mở đầu chung như “Theo quy định...” nếu không cần.
* Không kết luận dài dòng hoặc diễn giải lan man.
* Câu trả lời nên từ 2 đến 4 câu.
* Chỉ dùng gạch đầu dòng nếu câu hỏi yêu cầu liệt kê nhiều điều kiện hoặc nhiều mức xử lý.
* Mục tiêu tối đa khoảng 80 từ. Nếu phải chọn, ưu tiên đúng trọng tâm hơn là đầy đủ quá mức.

Trường hợp không đủ thông tin:

* Nếu evidence/candidates không có nội dung đủ để trả lời, trả lời ngắn gọn rằng chưa có đủ thông tin để xác định.
* Không tự suy đoán ngoài nội dung được cung cấp.

Chỉ trả về JSON đúng format:
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
            raise ValueError("Reasoner khong tra ve noi dung cau tra loi.")
        return AnswerDraft(answer=answer)
