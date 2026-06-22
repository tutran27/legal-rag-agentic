import json
from typing import Any

from src.schema.agent_schemas import (
    AnswerDraft,
    Evidence,
    LegalUnderstanding,
)


REASONER_MAX_TOKENS = 224
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
Ban la Reasoner Agent cua he thong hoi dap phap luat Viet Nam.

Nhiem vu:
- Doc selected evidence va tra loi dung trong tam cau hoi.
- Chi lay cac y khop truc tiep voi cau hoi.

Quy tac:
- Chi dung thong tin co trong evidence, khong them kien thuc ngoai.
- Uu tien tra loi thang vao dieu nguoi dung hoi.
- Giu du y quan trong nhu dieu kien, doi tuong, ngoai le, thu tuc, muc xu ly neu chung lien quan truc tiep.
- Gop y trung de cau tra loi ngan hon.
- Khong nhac den dieu luat, so van ban, evidence, retrieval, agent hay he thong.
- Khong viet cau mo dau, ket luan chung hoac dien giai dai.
- Cau tra loi nen tu 1 den 3 cau; chi dung gach dau dong neu that su can.
- Muc tieu toi da khoang 80 tu. Neu phai chon, uu tien dung trong tam hon la dien giai nhieu.

Chi tra ve JSON:
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
