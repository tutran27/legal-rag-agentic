import re
import unicodedata
from typing import Any

from src.schema.agent_schemas import (
    PlanningResult,
    QueryPlan,
    RetrievalPlan,
    SearchQuery,
)


QUERY_PLAN_MAX_TOKENS = 256
DEFAULT_CONTEXT = "đối tượng áp dụng căn cứ pháp lý phạm vi điều kiện thực hiện"

CONTEXT_RULES = (
    (("điều kiện", "tiêu chí", "hưởng", "hỗ trợ"), "điều kiện được hỗ trợ theo quy định pháp luật"),
    (("hồ sơ", "thủ tục", "đăng ký", "nộp"), "hồ sơ và thủ tục thực hiện theo quy định pháp luật"),
    (("nghĩa vụ", "trách nhiệm", "phải"), "nghĩa vụ và trách nhiệm theo quy định pháp luật"),
    (("xử phạt", "phạt", "vi phạm"), "hành vi vi phạm và mức xử phạt theo quy định pháp luật"),
    (("là gì", "khái niệm", "định nghĩa"), "khái niệm và tiêu chí xác định theo quy định pháp luật"),
    (("quyền", "được hưởng", "chính sách"), "quyền lợi và chính sách được hưởng theo quy định pháp luật"),
)

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
    "luat": "luật",
    "nghi dinh": "nghị định",
    "thong tu": "thông tư",
    "quyet dinh": "quyết định",
}

QUERY_PLANNER_PROMPT = """
Bạn là Query Planner Agent cho hệ thống hỏi đáp pháp luật Việt Nam.

Nhiệm vụ:
1. Tạo đúng 3 query phục vụ retrieval: original, legal_rewrite, keyword.
2. Tạo filter định danh nếu câu hỏi cung cấp thông tin chắc chắn.
3. Toàn bộ nội dung trả về phải dùng tiếng Việt có dấu.

Quy tắc query:
- original giữ nguyên câu hỏi.
- legal_rewrite là cụm truy vấn pháp lý ngắn, không có dấu hỏi.
- keyword là cụm từ khóa BM25 ngắn, không trùng legal_rewrite.
- Không dùng tiếng Anh cho nhãn nghiệp vụ trong text/reason.
- Không dùng slug, snake_case, kebab-case, dấu "_" hoặc tiếng Việt không dấu.

Quy tắc filter:
- Chỉ đặt doc_codes khi câu hỏi nêu rõ số hiệu văn bản.
- doc_types, domains, sectors phải là tiếng Việt có dấu, chữ thường.
- Không dùng giá trị kiểu "business", "tax", "condition_question", "doanh-nghiep".
- Chỉ đặt domain/sector/doc_type khi chắc chắn; nếu không chắc thì để [].
- Mặc định is_current=true.

Chỉ trả JSON theo schema:
{
  "plan": {
    "queries": [
      {"query_type": "original", "text": "..."},
      {"query_type": "legal_rewrite", "text": "..."},
      {"query_type": "keyword", "text": "..."}
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
    text = str(text or "").replace("_", " ")
    text = re.sub(r"[-]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_accents(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text)
    text = "".join(
        char for char in normalized if unicodedata.category(char) != "Mn"
    )
    return text.replace("đ", "d").replace("Đ", "D")


def _has_vietnamese_accent(text: str) -> bool:
    return _strip_accents(text or "") != (text or "")


def _normalize_taxonomy(value: str) -> str:
    text = _clean_text(value).lower()
    plain = _strip_accents(text)
    return TAXONOMY_ALIASES.get(plain, text)


def _unique_taxonomy(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        normalized = _normalize_taxonomy(value)
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _is_weak_query(query: SearchQuery | None, question_tokens: set[str]) -> bool:
    if query is None:
        return True
    text = query.text or ""
    return (
        "?" in text
        or "_" in text
        or "-" in text
        or not _has_vietnamese_accent(text)
        or len(_tokens(text) - question_tokens) < 1
    )


def _infer_context(question: str) -> str:
    plain_question = _strip_accents(question.lower())
    for keywords, context in CONTEXT_RULES:
        if any(_strip_accents(keyword) in plain_question for keyword in keywords):
            return context
    return DEFAULT_CONTEXT


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
        plan_data = data.get("plan", data)
        plan = QueryPlan(**plan_data)
        return PlanningResult(plan=self._normalize_plan(question, plan))

    def _normalize_plan(self, question: str, plan: QueryPlan) -> QueryPlan:
        queries = {query.query_type: query for query in plan.queries}
        question_tokens = _tokens(question)
        topic = self._topic(question, plan)
        context = _infer_context(question)

        legal_rewrite = queries.get("legal_rewrite")
        if _is_weak_query(legal_rewrite, question_tokens):
            legal_rewrite = SearchQuery(
                query_type="legal_rewrite",
                text=f"{topic} {context}",
                reason="Mở rộng câu hỏi thành truy vấn pháp lý có dấu",
            )

        keyword = queries.get("keyword")
        if (
            _is_weak_query(keyword, question_tokens)
            or keyword.text == legal_rewrite.text
        ):
            keyword = SearchQuery(
                query_type="keyword",
                text=f"{topic} {self._keyword_context(context)}",
                reason="Từ khóa pháp lý bám sát câu hỏi",
            )

        plan.queries = [
            SearchQuery(
                query_type="original",
                text=_clean_text(question),
                reason="Câu hỏi gốc",
            ),
            legal_rewrite.model_copy(
                update={
                    "text": _clean_text(legal_rewrite.text),
                    "reason": _clean_text(legal_rewrite.reason or ""),
                }
            ),
            keyword.model_copy(
                update={
                    "text": _clean_text(keyword.text),
                    "reason": _clean_text(keyword.reason or ""),
                }
            ),
        ]
        plan.filters = plan.filters.model_copy(
            update={
                "doc_types": _unique_taxonomy(plan.filters.doc_types),
                "domains": _unique_taxonomy(plan.filters.domains),
                "sectors": _unique_taxonomy(plan.filters.sectors),
            }
        )
        plan.retrieval = RetrievalPlan()
        return plan

    def _topic(self, question: str, plan: QueryPlan) -> str:
        values = [
            *plan.filters.domains,
            *plan.filters.sectors,
        ]
        topic = " ".join(_unique_taxonomy(values)).strip()
        if topic and _has_vietnamese_accent(topic):
            return topic
        return _clean_text(question.rstrip(" ?"))

    @staticmethod
    def _keyword_context(context: str) -> str:
        keyword_context = (
            context.replace("theo quy định pháp luật", "")
            .replace("được ", "")
            .strip()
        )
        if "điều kiện" in context:
            return f"tiêu chí {keyword_context}"
        return keyword_context
