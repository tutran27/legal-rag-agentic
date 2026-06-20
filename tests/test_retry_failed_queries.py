import importlib

from src.schema.agent_schemas import TestQuestion as QuestionSchema


retry_script = importlib.import_module("scripts.07_retry_failed_queries")


def test_get_retry_questions_keeps_input_order():
    questions = [
        QuestionSchema(id=3, question="Câu 3"),
        QuestionSchema(id=1, question="Câu 1"),
        QuestionSchema(id=2, question="Câu 2"),
    ]
    errors = [{"id": 2}, {"id": 3}, {"id": 999}]

    selected = retry_script.get_retry_questions(questions, errors)

    assert [item.id for item in selected] == [3, 2]


def test_update_error_removes_success_and_replaces_failure():
    question = QuestionSchema(id=2, question="Câu 2")
    errors = [{"id": 1, "error": "old"}, {"id": 2, "error": "old"}]

    successful = retry_script.update_error(errors, question, error=None)
    failed = retry_script.update_error(errors, question, ValueError("new"))

    assert successful == [{"id": 1, "error": "old"}]
    assert failed[-1] == {
        "id": 2,
        "question": "Câu 2",
        "error": "new",
    }
