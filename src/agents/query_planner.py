import json
from typing import Any

from src.schema.agent_schemas import LegalUnderstanding, QueryPlan, SearchQuery


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

Tạo các truy vấn ngắn để tìm đúng nội dung điều luật. Không trả lời câu hỏi.

Quy tắc:
- Luôn có một query_type "original" giữ nguyên câu hỏi.
- Tạo tối đa 3 query: original, legal_rewrite và keyword.
- keyword chỉ chứa thuật ngữ pháp lý quan trọng.
- Chỉ đặt doc_codes khi câu hỏi nêu rõ số hiệu văn bản.
- Chỉ đặt doc_types, domains, sectors khi chắc chắn.
- is_current=true nếu câu hỏi hỏi pháp luật hiện hành.
- Dense và sparse là retrieval chính.
- ColBERT dùng để rerank sau fusion.
- Graph dùng để mở rộng từ các văn bản đã tìm được.

Chỉ trả về JSON theo schema:
{
  "queries": [
    {
      "query_type": "original | legal_rewrite | keyword",
      "text": "...",
      "reason": "..."
    }
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
    "use_bm25": false,
    "use_dense": true,
    "use_sparse": true,
    "use_colbert": true,
    "use_graph": true,
    "use_summary": false,
    "top_k_exact": 20,
    "top_k_bm25": 50,
    "top_k_dense": 50,
    "top_k_sparse": 50,
    "top_k_colbert": 20,
    "top_k_graph": 30,
    "top_k_summary": 0
  }
}
"""
        context = {
            "question": question,
            "understanding": understanding.model_dump(),
        }
        data = self.llm.call_llm_json(
            query=json.dumps(context, ensure_ascii=False),
            system_prompt=prompt,
            max_new_tokens=1536,
            temperature=0.1,
        )

        plan = QueryPlan(**data)
        if not any(query.query_type == "original" for query in plan.queries):
            plan.queries.insert(
                0,
                SearchQuery(
                    query_type="original",
                    text=question,
                    reason="Câu hỏi gốc",
                ),
            )
        return plan


if __name__ == "__main__":
    from src.agents.legal_understanding import LegalUnderstandingAgent
    from src.generation.llm_service import GroqLLMClient
    
    import json
    
    llm = GroqLLMClient()
    
    legal_understanding_agent = LegalUnderstandingAgent(llm)
    query_planner_agent = QueryPlannerAgent(llm)
    
    query="Doanh nghiệp SME có thể có bao nhiêu người làm việc?"
    understanding = legal_understanding_agent.run(query)
    print("Understanding:\n", json.dumps(understanding.model_dump(), ensure_ascii=False, indent=2))
    plan = query_planner_agent.run(query, understanding)
    print("Query Plan:\n", json.dumps(plan.model_dump(), ensure_ascii=False, indent=2))
