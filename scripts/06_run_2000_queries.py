import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import TypeAdapter, ValidationError


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.schema.agent_schemas import SubmissionItem, TestQuestion
from src.submission.build_results import write_results


DEFAULT_INPUT = "R2AIStage1DATA.json"
DEFAULT_OUTPUT = "results.json"
DEFAULT_ERRORS = "inference_errors.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy batch inference lớn, giữ model sống trong một process."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--errors", default=DEFAULT_ERRORS)
    parser.add_argument("--limit", type=int, default=2000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--llm", choices=["groq", "endpoint", "local"], default=None)
    parser.add_argument("--local-model", default=None)
    return parser.parse_args()


def load_questions(path: str | Path, limit: int | None) -> list[TestQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("questions") or data.get("data") or data.get("items")
    if not isinstance(data, list):
        raise ValueError("Input phải là list hoặc dict có questions/data/items.")
    questions = TypeAdapter(list[TestQuestion]).validate_python(data)
    return questions[:limit] if limit else questions


def has_valid_citation_format(item: SubmissionItem) -> bool:
    for citation in item.relevant_docs:
        parts = citation.split("|", 1)
        if len(parts) != 2 or not all(parts):
            return False
    for citation in item.relevant_articles:
        parts = citation.split("|", 2)
        if len(parts) != 3 or not all(parts):
            return False
    return True


def load_results(path: str | Path) -> dict[int, SubmissionItem]:
    result_path = Path(path)
    if not result_path.exists():
        return {}
    content = result_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return {}
    try:
        data = json.loads(content)
    except json.JSONDecodeError as error:
        raise ValueError(f"Output JSON hỏng hoặc chưa ghi xong: {result_path}") from error
    if not isinstance(data, list):
        raise ValueError("Output hiện có phải là list JSON.")
    items = TypeAdapter(list[SubmissionItem]).validate_python(data)
    valid_items = {
        item.id: item
        for item in items
        if has_valid_citation_format(item)
    }
    invalid_count = len(items) - len(valid_items)
    if invalid_count:
        print(
            f"Bỏ qua {invalid_count} kết quả dùng format citation cũ; "
            "các query này sẽ được chạy lại."
        )
    return valid_items


def load_errors(path: str | Path) -> list[dict]:
    error_path = Path(path)
    if not error_path.exists():
        return []
    content = error_path.read_text(encoding="utf-8-sig").strip()
    if not content:
        return []
    data = json.loads(content)
    return data if isinstance(data, list) else []


def write_errors(errors: list[dict], path: str | Path) -> None:
    error_path = Path(path)
    error_path.parent.mkdir(parents=True, exist_ok=True)
    error_path.write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def write_ordered_results(
    questions: list[TestQuestion],
    results: dict[int, SubmissionItem],
    output_path: str | Path,
) -> None:
    ordered = [results[item.id] for item in questions if item.id in results]
    write_results(ordered, output_path)


def fallback_submission(question: TestQuestion, error: Exception) -> SubmissionItem:
    return SubmissionItem(
        id=question.id,
        question=question.question,
        answer=(
            "Chưa có đủ căn cứ pháp lý liên quan để trả lời câu hỏi."
        ),
        relevant_docs=[],
        relevant_articles=[],
    )


def build_pipeline(args: argparse.Namespace):
    from src.pipeline import InferencePipeline
    from src.generation.endpoint import create_llm_client

    llm = create_llm_client(args.llm, args.local_model)
    return InferencePipeline(llm=llm, verbose=False)


def run_batch(args: argparse.Namespace) -> None:
    questions = load_questions(args.input, args.limit)
    results = load_results(args.output) if args.resume else {}
    errors = load_errors(args.errors) if args.resume else []
    pending = [item for item in questions if item.id not in results]
    write_ordered_results(questions, results, args.output)
    write_errors(errors, args.errors)

    print(
        f"Tổng câu hỏi: {len(questions)} | đã có: {len(results)} | "
        f"cần chạy: {len(pending)}"
    )
    if not pending:
        write_ordered_results(questions, results, args.output)
        return

    init_started = time.perf_counter()
    pipeline = build_pipeline(args)
    print(f"Khởi tạo model/pipeline: {time.perf_counter() - init_started:.1f}s")

    started = time.perf_counter()
    try:
        for index, question in enumerate(pending, start=1):
            query_started = time.perf_counter()
            status = "OK"
            try:
                output = pipeline.run(question.question, question_id=question.id)
                results[question.id] = output.submission
            except Exception as error:
                status = "ERROR"
                errors = [
                    item for item in errors if item.get("id") != question.id
                ]
                errors.append(
                    {
                        "id": question.id,
                        "question": question.question,
                        "error": str(error),
                    }
                )
                print(f"[ERROR DETAIL] id={question.id}: {error}")

            write_ordered_results(questions, results, args.output)
            write_errors(errors, args.errors)

            elapsed = time.perf_counter() - query_started
            done = len(results)
            print(
                f"[{status}] {index}/{len(pending)} | "
                f"id={question.id} | {elapsed:.1f}s | done={done}/{len(questions)}"
            )
    finally:
        pipeline.close()

    total = time.perf_counter() - started
    print(
        f"Hoàn tất: {len(results)}/{len(questions)} kết quả | "
        f"errors={len(errors)} | total={total:.1f}s"
    )


def main() -> None:
    args = parse_args()
    try:
        run_batch(args)
    except ValidationError as error:
        raise SystemExit(f"Input/output không đúng schema: {error}") from error


if __name__ == "__main__":
    main()
