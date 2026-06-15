from jsonschema import Draft202012Validator

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
