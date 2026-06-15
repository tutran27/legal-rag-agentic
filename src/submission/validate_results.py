import json
from pathlib import Path

from jsonschema import Draft202012Validator
from pydantic import ValidationError

from src.schema.agent_schemas import Evidence, SubmissionItem


def validate_submission_item(
    item: SubmissionItem,
    evidence: list[Evidence],
) -> None:
    schema = SubmissionItem.model_json_schema()
    errors = sorted(
        Draft202012Validator(schema).iter_errors(item.model_dump()),
        key=lambda error: list(error.path),
    )
    if errors:
        message = "; ".join(error.message for error in errors)
        raise ValueError(f"Submission không đúng JSON Schema: {message}")

    valid_docs = set()
    valid_articles = set()
    for source in evidence:
        doc_code = source.doc_code or source.metadata.get("doc_code")
        article = source.article or source.metadata.get("article")
        if doc_code:
            valid_docs.add(str(doc_code))
        if article:
            citation = (
                f"{doc_code} - {article}"
                if doc_code
                else str(article)
            )
            valid_articles.add(citation)

    invalid_docs = set(item.relevant_docs) - valid_docs
    invalid_articles = set(item.relevant_articles) - valid_articles
    if invalid_docs or invalid_articles:
        raise ValueError(
            "Citation không thuộc selected evidence: "
            f"docs={sorted(invalid_docs)}, "
            f"articles={sorted(invalid_articles)}"
        )


def load_and_validate_results(
    input_path: str | Path = "results.json",
) -> list[SubmissionItem]:
    path = Path(input_path)
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Không đọc được {path}: {error}") from error

    if not isinstance(data, list):
        raise ValueError("results.json phải là một danh sách.")

    try:
        return [SubmissionItem.model_validate(item) for item in data]
    except ValidationError as error:
        raise ValueError(f"results.json không đúng schema: {error}") from error
