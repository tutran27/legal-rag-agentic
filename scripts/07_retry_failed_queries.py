import argparse
import importlib
import sys
import time
from pathlib import Path

from pydantic import ValidationError


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

batch = importlib.import_module("scripts.06_run_2000_queries")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy lại các ID trong error JSON và cập nhật results.json."
    )
    parser.add_argument("--input", default=batch.DEFAULT_INPUT)
    parser.add_argument("--output", default=batch.DEFAULT_OUTPUT)
    parser.add_argument("--errors", default=batch.DEFAULT_ERRORS)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--llm",
        choices=["groq", "endpoint", "local"],
        default=None,
    )
    parser.add_argument("--local-model", default=None)
    return parser.parse_args()


def get_retry_questions(questions, errors, limit=None):
    error_ids = {
        item.get("id")
        for item in errors
        if isinstance(item, dict) and isinstance(item.get("id"), int)
    }
    retry_questions = [item for item in questions if item.id in error_ids]
    return retry_questions[:limit] if limit else retry_questions


def update_error(errors, question, error):
    remaining = [
        item
        for item in errors
        if not isinstance(item, dict) or item.get("id") != question.id
    ]
    if error is not None:
        remaining.append(
            {
                "id": question.id,
                "question": question.question,
                "error": str(error),
            }
        )
    return remaining


def run(args: argparse.Namespace) -> None:
    questions = batch.load_questions(args.input, limit=None)
    results = batch.load_results(args.output)
    errors = batch.load_errors(args.errors)
    retry_questions = get_retry_questions(questions, errors, args.limit)

    known_ids = {item.id for item in questions}
    unknown_ids = sorted(
        {
            item.get("id")
            for item in errors
            if isinstance(item, dict)
            and isinstance(item.get("id"), int)
            and item.get("id") not in known_ids
        }
    )
    if unknown_ids:
        print(f"Bỏ qua ID không có trong input: {unknown_ids}")

    print(
        f"Lỗi trong file: {len(errors)} | "
        f"ID hợp lệ cần chạy lại: {len(retry_questions)}"
    )
    if not retry_questions:
        batch.write_ordered_results(questions, results, args.output)
        return

    pipeline = batch.build_pipeline(args)
    try:
        for index, question in enumerate(retry_questions, start=1):
            started = time.perf_counter()
            try:
                output = pipeline.run(
                    question.question,
                    question_id=question.id,
                )
                results[question.id] = output.submission
                errors = update_error(errors, question, error=None)
                status = "OK"
            except Exception as error:
                errors = update_error(errors, question, error)
                status = "ERROR"
                print(f"[ERROR DETAIL] id={question.id}: {error}")

            batch.write_ordered_results(questions, results, args.output)
            batch.write_errors(errors, args.errors)
            elapsed = time.perf_counter() - started
            print(
                f"[{status}] {index}/{len(retry_questions)} | "
                f"id={question.id} | {elapsed:.1f}s | errors={len(errors)}"
            )
    finally:
        pipeline.close()


def main() -> None:
    args = parse_args()
    try:
        run(args)
    except ValidationError as error:
        raise SystemExit(f"Input/output không đúng schema: {error}") from error


if __name__ == "__main__":
    main()
