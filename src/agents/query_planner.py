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


QUERY_PLAN_MAX_TOKENS =512

INTENT_CONTEXT = {
    "condition": "điều kiện tiêu chí đối tượng được hỗ trợ",
    "procedure": "hồ sơ thủ tục trình tự thực hiện",
    "obligation": "nghĩa vụ trách nhiệm phải thực hiện",
    "penalty": "hành vi vi phạm mức xử phạt biện pháp khắc phục",
    "definition": "khái niệm tiêu chí xác định",
    "rights": "quyền lợi điều kiện được hưởng",
}
DEFAULT_CONTEXT = "quy định pháp luật điều kiện trách nhiệm thủ tục"

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
    "giao duc": "giáo dục",
    "dat dai": "đất đai",
}

QUERY_PLANNER_PROMPT = """
Bạn là Query Planner cho hệ thống Legal RAG Việt Nam.

Nhiệm vụ: chỉ tạo kế hoạch truy vấn retrieval. Không trả phần understanding.

Quy tắc bắt buộc:
- Trả đúng 2 query: original và legal_rewrite.
- original giữ nguyên câu hỏi người dùng.
- legal_rewrite viết lại thành một mệnh đề/cụm truy vấn pháp lý ngắn, đúng tiếng Việt có dấu.
- legal_rewrite phải bám sát câu gốc, không mở rộng sang chủ đề chung nếu câu gốc có hành vi cụ thể.
- Có thể đảo vị trí chủ thể - hành vi - hậu quả, nhưng không làm mất ý chính.
- Cấm tiếng Việt không dấu, slug, gạch dưới, từ khóa rời rạc hoặc câu hỏi có dấu hỏi.
- Không tạo keyword query.
- Filter doc_codes chỉ đặt khi câu hỏi nêu rõ số hiệu văn bản.
- Filter doc_types, domains, sectors phải là tiếng Việt thường, có dấu, không slug.
- Chỉ đặt domain/sector khi chắc chắn; nếu không chắc thì để [].
- Mặc định is_current=true.

Ví dụ:
Câu hỏi: Nếu công ty giữ bản chính bằng cấp của nhân viên khi ký hợp đồng thì xử lý thế nào?
legal_rewrite: công ty giữ bản chính bằng cấp của người lao động khi giao kết hợp đồng lao động bị xử lý và khắc phục

Chỉ trả JSON:
{
  "plan": {
    "queries": [
      {"query_type": "original", "text": "...", "reason": "Câu hỏi gốc"},
      {"query_type": "legal_rewrite", "text": "...", "reason": "Viết lại pháp lý bám sát câu hỏi"}
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
    alias_key = _strip_accents(text)
    return TAXONOMY_ALIASES.get(alias_key, text)


def _unique_taxonomy(values: list[str]) -> list[str]:
    result = []
    seen = set()
    for value in values:
        normalized = _normalize_taxonomy(value)
        if normalized and normalized not in seen:
            result.append(normalized)
            seen.add(normalized)
    return result


def _is_bad_rewrite(query: SearchQuery | None, question: str) -> bool:
    if query is None:
        return True
    text = _clean_text(query.text)
    return (
        not text
        or "?" in text
        or "_" in text
        or not _has_vietnamese_accent(text)
        or _strip_accents(text) == _strip_accents(question)
    )


def _guess_context(question: str) -> str:
    normalized = question.lower()
    if any(term in normalized for term in ("điều kiện", "tiêu chí")):
        return INTENT_CONTEXT["condition"]
    if any(term in normalized for term in ("thủ tục", "hồ sơ", "trình tự")):
        return INTENT_CONTEXT["procedure"]
    if any(term in normalized for term in ("nghĩa vụ", "trách nhiệm", "phải")):
        return INTENT_CONTEXT["obligation"]
    if any(term in normalized for term in ("xử phạt", "xử lý", "mức phạt")):
        return INTENT_CONTEXT["penalty"]
    if any(term in normalized for term in ("là gì", "khái niệm")):
        return INTENT_CONTEXT["definition"]
    if any(term in normalized for term in ("quyền", "được hưởng")):
        return INTENT_CONTEXT["rights"]
    return DEFAULT_CONTEXT


class QueryPlannerAgent:
    def __init__(self, llm: Any):
        self.llm = llm

    def run(self, question: str) -> PlanningResult:
        data = self.llm.call_llm_json(
            query=question,
            system_prompt=QUERY_PLANNER_PROMPT,
            retry_stage="query_planning",
            max_new_tokens=QUERY_PLAN_MAX_TOKENS,
            temperature=0.0,
        )
        result = PlanningResult(**data)
        plan = self._normalize_plan(question, result.plan)
        return result.model_copy(update={"plan": plan})

    def _normalize_plan(self, question: str, plan: QueryPlan) -> QueryPlan:
        queries = {query.query_type: query for query in plan.queries}
        rewrite = queries.get("legal_rewrite")
        if _is_bad_rewrite(rewrite, question):
            rewrite = SearchQuery(
                query_type="legal_rewrite",
                text=f"{_clean_text(question.rstrip(' ?'))} {_guess_context(question)}",
                reason="Viết lại pháp lý bám sát câu hỏi",
            )

        normalized_queries = [
            SearchQuery(
                query_type="original",
                text=_clean_text(question),
                reason="Câu hỏi gốc",
            ),
            rewrite.model_copy(update={"text": _clean_text(rewrite.text)}),
        ]
        filters = plan.filters.model_copy(
            update={
                "domains": _unique_taxonomy(plan.filters.domains),
                "sectors": _unique_taxonomy(plan.filters.sectors),
            }
        )
        return plan.model_copy(
            update={
                "queries": normalized_queries,
                "filters": filters,
                "retrieval": RetrievalPlan(),
            }
        )


def make_empty_understanding() -> LegalUnderstanding:
    return LegalUnderstanding()
