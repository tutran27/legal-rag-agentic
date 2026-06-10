from dataclasses import dataclass, field


@dataclass
class LegalAgentState:
    id: int
    question: str

    domain: str | None = None                                            # phân loại domain: lao động / thuế / doanh nghiệp / hợp đồng / BHXH
    intent: str | None = None
    search_queries: list[str] = field(default_factory=list)              # danh sách các query được dùng để tìm kiếm

    candidates: list[dict] = field(default_factory=list)                  # danh sách các candidate được dùng để trả lời câu hỏi
    selected_evidence: list[dict] = field(default_factory=list)          # danh sách các evidence được chọn để trả lời câu hỏi

    answer: str | None = None                                            # trả lời câu hỏi được chọn từ candidate được chọn
    relevant_docs: list[str] = field(default_factory=list)              # danh sách các document liên quan đến câu hỏi
    relevant_articles: list[str] = field(default_factory=list)          # danh sách các article liên quan đến câu hỏi

    errors: list[str] = field(default_factory=list)                      # danh sách các lỗi xảy ra trong quá trình xử lý câu hỏi