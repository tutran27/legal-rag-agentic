import json

from src.schema.agent_schemas import SubmissionItem
from src.submission.build_results import write_results


def test_write_results(tmp_path):
    item = SubmissionItem(
        id=1,
        question="Câu hỏi",
        answer="Câu trả lời",
        relevant_docs=["04/2017/QH14"],
        relevant_articles=["04/2017/QH14 - Điều 5"],
    )

    output = write_results([item], tmp_path / "results.json")
    data = json.loads(output.read_text(encoding="utf-8"))

    assert data == [item.model_dump()]
