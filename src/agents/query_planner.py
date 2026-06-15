import json
import re
from typing import Any

from src.schema.agent_schemas import LegalUnderstanding, QueryPlan, SearchQuery


LEGAL_CONTEXTS = {
    "condition_question": (
        "tiêu chí và điều kiện được hỗ trợ"
    ),
    "procedure_question": (
        "hồ sơ và thủ tục thực hiện"
    ),
    "obligation_question": (
        "nghĩa vụ và trách nhiệm thực hiện"
    ),
    "penalty_question": (
        "hành vi vi phạm và mức xử phạt"
    ),
    "definition_question": (
        "khái niệm và tiêu chí xác định"
    ),
    "rights_question": (
        "quyền lợi và điều kiện được hưởng"
    ),
}


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


class QueryPlannerAgent:
    def __init__(self, llm: Any):
        self.llm = llm

    def run(
        self,
        question: str,
        understanding: LegalUnderstanding,
    ) -> QueryPlan:
        prompt = """
Bạn là Query Planner cho hệ thống Legal RAG Việt Nam.

Tạo đúng 3 truy vấn có mục đích khác nhau:
1. original: giữ nguyên câu hỏi.
2. legal_rewrite: lệnh đề pháp lý mở rộng ngữ cảnh.
3. keyword: tập hợp thuật ngữ pháp lý để tìm bằng BM25.

Yêu cầu bắt buộc:
- Không được tạo ba cách diễn đạt cùng một nội dung.
- legal_rewrite không phải câu hỏi, không có dấu "?" hoặc từ nghi vấn.
- legal_rewrite phải ngắn gọn, bám sát chủ thể và ý định của original.
- Chỉ thêm 1-2 khái niệm pháp lý trực tiếp liên quan, không mở rộng sang
  ngoại lệ, hồ sơ hoặc thủ tục nếu câu hỏi không đề cập.
- keyword là cụm từ khóa ngắn, giữ chủ thể chính và ý định pháp lý.

Ví dụ:
original:
"Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ?"

legal_rewrite:
"Tiêu chí và điều kiện doanh nghiệp nhỏ và vừa được hỗ trợ"

keyword:
"doanh nghiệp nhỏ và vừa tiêu chí điều kiện hỗ trợ"

Chỉ đặt doc_codes khi câu hỏi nêu rõ số hiệu văn bản.
Chỉ đặt doc_types, domains, sectors khi chắc chắn.
is_current=true nếu hỏi pháp luật hiện hành.

Chỉ trả JSON theo schema:
{
  "queries": [
    {"query_type": "original", "text": "...", "reason": "..."},
    {"query_type": "legal_rewrite", "text": "...", "reason": "..."},
    {"query_type": "keyword", "text": "...", "reason": "..."}
  ],
  "filters": {
    "doc_codes": [],
    "doc_types": [],
    "domains": [],
    "sectors": [],
    "is_current": true
  },
  "retrieval": {
    "use_exact": true,
    "use_bm25": true,
    "use_dense": true,
    "use_sparse": true,
    "use_colbert": true,
    "use_cross_encoder": true,
    "use_graph": true,
    "use_context": true,
    "use_summary": false,
    "top_k_exact": 50,
    "top_k_bm25": 100,
    "top_k_dense": 100,
    "top_k_sparse": 100,
    "top_k_colbert": 60,
    "top_k_cross_encoder": 40,
    "top_k_graph": 50,
    "top_k_summary": 30
  }
}
"""
        data = self.llm.call_llm_json(
            query=json.dumps(
                {
                    "question": question,
                    "understanding": understanding.model_dump(),
                },
                ensure_ascii=False,
            ),
            system_prompt=prompt,
            max_new_tokens=1536,
            temperature=0.1,
        )

        plan = QueryPlan(**data)
        queries = {query.query_type: query for query in plan.queries}
        original_tokens = _tokens(question)

        topic = " ".join(understanding.legal_entities).strip()
        if not topic:
            topic = understanding.domain or question.rstrip(" ?")
        context = LEGAL_CONTEXTS.get(
            understanding.intent or "",
            "đối tượng áp dụng căn cứ pháp lý phạm vi điều kiện thực hiện",
        )

        legal_rewrite = queries.get("legal_rewrite")
        legal_is_weak = (
            legal_rewrite is None
            or "?" in legal_rewrite.text
            or len(_tokens(legal_rewrite.text) - original_tokens) < 1
        )
        if legal_is_weak:
            legal_rewrite = SearchQuery(
                query_type="legal_rewrite",
                text=f"{topic} {context}",
                reason="Mở rộng theo các khái niệm pháp lý liên quan",
            )

        keyword = queries.get("keyword")
        keyword_is_weak = (
            keyword is None
            or len(_tokens(keyword.text) - original_tokens) < 1
            or keyword.text == legal_rewrite.text
        )
        if keyword_is_weak:
            keyword = SearchQuery(
                query_type="keyword",
                text=f"{topic} {context.replace(' và ', ' ')}",
                reason="Từ khóa pháp lý bám sát câu hỏi",
            )

        plan.queries = [
            SearchQuery(
                query_type="original",
                text=question,
                reason="Câu hỏi gốc",
            ),
            legal_rewrite,
            keyword,
        ]
        return plan


if __name__ == "__main__":
    from src.agents.legal_understanding import LegalUnderstandingAgent
    from src.generation.llm_service import GroqLLMClient

    llm = GroqLLMClient()
    question = "Doanh nghiệp SME có thể có bao nhiêu người làm việc?"
    understanding = LegalUnderstandingAgent(llm).run(question)
    plan = QueryPlannerAgent(llm).run(question, understanding)
    print(json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))
