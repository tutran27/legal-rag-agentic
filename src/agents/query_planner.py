import re
import unicodedata
from typing import Any

from src.schema.agent_schemas import (
    LegalUnderstanding,
    PlanningResult,
    QueryPlan,
    RetrievalPlan,
    SearchQuery,
)


QUERY_PLAN_MAX_TOKENS = 512

INTENT_CONTEXT = {
    "condition_question": "điều kiện được hỗ trợ theo quy định pháp luật",
    "procedure_question": "hồ sơ và thủ tục thực hiện theo quy định pháp luật",
    "obligation_question": "nghĩa vụ và trách nhiệm thực hiện theo quy định pháp luật",
    "penalty_question": "hành vi vi phạm và mức xử phạt theo quy định pháp luật",
    "definition_question": "khái niệm và tiêu chí xác định theo quy định pháp luật",
    "rights_question": "quyền lợi và điều kiện được hưởng theo quy định pháp luật",
}
DEFAULT_CONTEXT = "đối tượng áp dụng căn cứ pháp lý phạm vi điều kiện thực hiện"

TAXONOMY_ALIASES = {
    "doanh nghiep": "doanh nghiệp",
    "hop tac xa": "hợp tác xã",
    "doanh nghiep hop tac xa": "doanh nghiệp, hợp tác xã",
    "thue": "thuế",
    "phi le phi": "phí, lệ phí",
    "lao dong": "lao động",
    "bao hiem xa hoi": "bảo hiểm xã hội",
    "ke toan": "kế toán",
    "hop dong": "hợp đồng",
    "xu phat": "xử phạt",
    "dau tu": "đầu tư",
    "ke hoach va dau tu": "kế hoạch và đầu tư",
}

QUERY_PLANNER_PROMPT = """
Bạn là agent phân tích câu hỏi và tạo truy vấn tìm kiếm cho Legal RAG Việt Nam.

Nhiệm vụ:
1. Phân tích câu hỏi.
2. Tạo đúng 3 query: original, legal_rewrite, keyword.
3. Tạo filter định danh nếu chắc chắn.

Quy tắc:
- original giữ nguyên câu hỏi.
- legal_rewrite là cụm truy vấn pháp lý ngắn, có dấu tiếng Việt, không có dấu hỏi.
- keyword là cụm từ khóa BM25 ngắn, có dấu tiếng Việt, không trùng legal_rewrite.
- Không dùng slug, dấu "_" hoặc tiếng Việt không dấu.
- Chỉ đặt doc_codes khi câu hỏi nêu rõ số hiệu văn bản.
- Chỉ đặt domain/sector khi chắc chắn; mặc định is_current=true.

Chỉ trả JSON:
{
  "understanding": {
    "domain": "...",
    "intent": "condition_question | obligation_question | procedure_question | penalty_question | definition_question | rights_question | document_lookup | other",
    "answer_type": "...",
    "legal_entities": ["..."],
    "likely_docs": ["..."],
    "sub_questions": ["..."],
    "missing_facts": ["..."],
    "time_context": "hiện hành",
    "need_effective_check": true
  },
  "plan": {
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
    }
  }
}
"""


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower(), flags=re.UNICODE))


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").replace("_", " ")).strip()


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return text.replace("đ", "d").replace("Đ", "D")


def _has_vietnamese_accent(text: str) -> bool:
    text = text or ""
    return _strip_accents(text) != text


def _normalize_taxonomy(value: str) -> str:
    text = re.sub(r"[_-]+", " ", str(value or "")).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return TAXONOMY_ALIASES.get(_strip_accents(text), text)


def _unique_taxonomy(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        value = _normalize_taxonomy(value)
        if value and value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _is_weak_query(query: SearchQuery | None, question_tokens: set[str]) -> bool:
    if query is None:
        return True
    text = query.text
    return (
        "?" in text
        or "_" in text
        or not _has_vietnamese_accent(text)
        or len(_tokens(text) - question_tokens) < 1
    )


class QueryPlannerAgent:
    def __init__(self, llm: Any):
        self.llm = llm

    def run(self, question: str) -> PlanningResult:
        data = self.llm.call_llm_json(
            query=question,
            system_prompt=QUERY_PLANNER_PROMPT,
            max_new_tokens=QUERY_PLAN_MAX_TOKENS,
            temperature=0.1,
        )
        result = PlanningResult(**data)
        plan = self._normalize_plan(question, result.understanding, result.plan)
        return result.model_copy(update={"plan": plan})

    def _normalize_plan(
        self,
        question: str,
        understanding: LegalUnderstanding,
        plan: QueryPlan,
    ) -> QueryPlan:
        queries = {query.query_type: query for query in plan.queries}
        question_tokens = _tokens(question)
        topic = self._topic(question, understanding)
        context = INTENT_CONTEXT.get(understanding.intent or "", DEFAULT_CONTEXT)

        legal_rewrite = queries.get("legal_rewrite")
        if _is_weak_query(legal_rewrite, question_tokens):
            legal_rewrite = SearchQuery(
                query_type="legal_rewrite",
                text=f"{topic} {context}",
                reason="Mở rộng theo khái niệm pháp lý liên quan",
            )

        keyword = queries.get("keyword")
        if (
            _is_weak_query(keyword, question_tokens)
            or keyword.text == legal_rewrite.text
        ):
            keyword = SearchQuery(
                query_type="keyword",
                text=f"{topic} {self._keyword_context(context, understanding)}",
                reason="Từ khóa pháp lý bám sát câu hỏi",
            )

        plan.queries = [
            SearchQuery(
                query_type="original",
                text=_clean_text(question),
                reason="Câu hỏi gốc",
            ),
            legal_rewrite.model_copy(
                update={"text": _clean_text(legal_rewrite.text)}
            ),
            keyword.model_copy(update={"text": _clean_text(keyword.text)}),
        ]
        plan.filters = plan.filters.model_copy(
            update={
                "domains": _unique_taxonomy(plan.filters.domains),
                "sectors": _unique_taxonomy(plan.filters.sectors),
            }
        )
        plan.retrieval = RetrievalPlan()
        return plan

    def _topic(
        self,
        question: str,
        understanding: LegalUnderstanding,
    ) -> str:
        topic = " ".join(understanding.legal_entities).strip()
        topic = topic or understanding.domain or question.rstrip(" ?")
        topic = _clean_text(topic)
        return topic if _has_vietnamese_accent(topic) else _clean_text(question.rstrip(" ?"))

    @staticmethod
    def _keyword_context(
        context: str,
        understanding: LegalUnderstanding,
    ) -> str:
        keyword_context = (
            context.replace("theo quy định pháp luật", "")
            .replace("được ", "")
            .strip()
        )
        if understanding.intent == "condition_question":
            return f"tiêu chí {keyword_context}"
        return keyword_context
