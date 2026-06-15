import importlib
import json


inference_script = importlib.import_module("scripts.03_run_inference")


def test_load_questions(tmp_path):
    path = tmp_path / "questions.json"
    path.write_text(
        json.dumps(
            [{"id": 1, "question": "Câu hỏi pháp luật?"}],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    questions = inference_script.load_questions(path)

    assert questions[0].id == 1
    assert questions[0].question == "Câu hỏi pháp luật?"


def test_load_existing_results_returns_empty_for_missing_file(tmp_path):
    results = inference_script.load_existing_results(
        tmp_path / "results.json"
    )

    assert results == []
