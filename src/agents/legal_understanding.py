from src.schema.agent_schemas import LegalUnderstanding
from src.generation.endpoint import EndpointLLMClient


class LegalUnderstandingAgent:
    def __init__(self, llm: EndpointLLMClient):
        self.llm = llm

    def run(self, question: str) -> LegalUnderstanding:
        system_prompt = """
Bạn là Legal Understanding Agent cho hệ thống Legal RAG Việt Nam.

Nhiệm vụ:
Phân tích câu hỏi pháp lý của doanh nghiệp SME.

Chỉ trả về JSON hợp lệ, không giải thích ngoài JSON.

Schema:
{{
  "domain": "doanh nghiệp | thuế | lao động | bảo hiểm xã hội | hợp đồng | kế toán | xử phạt | khác",
  "intent": "condition_question | obligation_question | procedure_question | penalty_question | definition_question | rights_question | document_lookup | other",
  "answer_type": "điều kiện | nghĩa vụ | thủ tục | xử phạt | định nghĩa | quyền | tư vấn tình huống | khác",
  "legal_entities": ["..."],
  "likely_docs": ["..."],
  "sub_questions": ["..."],
  "missing_facts": ["..."],
  "time_context": "hiện hành hoặc thời điểm cụ thể nếu câu hỏi có nêu",
  "need_effective_check": true
}}
"""

        data = self.llm.call_llm_json(
            query=question,
            system_prompt=system_prompt,
            max_new_tokens=1024,
            temperature=0.1,
            enable_thinking=False,
        )

        return LegalUnderstanding(**data)
    
if __name__ == "__main__":
    import json
    
    llm = EndpointLLMClient()
    agent = LegalUnderstandingAgent(llm)
    result = agent.run("Doanh nghiệp SME có thể có bao nhiêu người làm việc?")
    data = result.model_dump() if hasattr(result, "model_dump") else result.dict()  
    print(json.dumps(data, ensure_ascii=False, indent=2))
