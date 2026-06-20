import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.pipeline import InferencePipeline
from src.generation.endpoint import create_llm_client
from src.schema.agent_schemas import InferenceResult, SubmissionItem, TestQuestion
from src.submission.build_results import write_results


DEFAULT_QUERY = (
    "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ "
    "theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Chạy inference 1 query hoặc một file nhiều query."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--question-id", type=int, default=1)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--llm",
        choices=["groq", "endpoint", "local"],
        default=None,
        help="Mặc định lấy theo LLM_BACKEND, nếu không có thì dùng endpoint.",
    )
    parser.add_argument(
        "--local-model",
        default=None,
        help="Model local khi dùng --llm local.",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def load_questions(path: str | Path) -> list[TestQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if isinstance(data, dict):
        data = data.get("questions") or data.get("data") or data.get("items")
    if not isinstance(data, list):
        raise ValueError("File input phải là list hoặc dict có questions/data/items.")
    try:
        return [TestQuestion.model_validate(item) for item in data]
    except ValidationError as error:
        raise ValueError(f"File input không đúng schema: {error}") from error


def load_existing_results(path: str | Path) -> list[SubmissionItem]:
    result_path = Path(path)
    if not result_path.exists():
        return []
    data = json.loads(result_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("File output hiện có phải là list.")
    return [SubmissionItem.model_validate(item) for item in data]


def print_result(result: InferenceResult, output_path: Path) -> None:
    print("================ SUBMISSION ==================")
    print(
        json.dumps(
            result.submission.model_dump(),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"Đã ghi kết quả: {output_path.resolve()}")
    print("================ LATENCY SUMMARY ==================")
    for name, elapsed in result.latencies.items():
        print(f"{name}: {elapsed:.3f}s")


def run_single(args: argparse.Namespace) -> None:
    print("================ INPUT ==================")
    print(f"Question ID: {args.question_id}")
    print(f"Query: {args.query}")
    print("================ PIPELINE INIT ==================")
    init_started = time.perf_counter()
    llm = create_llm_client(args.llm, args.local_model)
    pipeline = InferencePipeline(llm=llm, verbose=not args.quiet)
    print(f"Khởi tạo pipeline xong: {time.perf_counter() - init_started:.3f}s")
    try:
        print("================ RUN INFERENCE ==================")
        try:
            result = pipeline.run(args.query, question_id=args.question_id)
        except ValueError as error:
            print("================ INFERENCE FAILED ==================")
            print(str(error))
            return
    finally:
        print("================ PIPELINE CLOSE ==================")
        pipeline.close()
    print("================ INFERENCE DONE ==================")
    output_path = write_results([result.submission], args.output)
    print_result(result, output_path)


def run_file(args: argparse.Namespace) -> None:
    questions = load_questions(args.input)
    results = load_existing_results(args.output)
    completed_ids = {item.id for item in results}
    pending = [
        question
        for question in questions
        if question.id not in completed_ids
    ]
    print(
        f"Tổng câu hỏi: {len(questions)} | đã có: {len(results)} | "
        f"cần chạy: {len(pending)}"
    )
    if not pending:
        return

    llm = create_llm_client(args.llm, args.local_model)
    pipeline = InferencePipeline(llm=llm, verbose=not args.quiet)
    started = time.perf_counter()
    try:
        for offset in range(0, len(pending), args.batch_size):
            batch = pending[offset : offset + args.batch_size]
            print(
                f"Batch {offset // args.batch_size + 1}: "
                f"{offset + 1}-{offset + len(batch)}/{len(pending)}"
            )
            outputs = pipeline.run_many(
                [(item.id, item.question) for item in batch]
            )
            for question, output in zip(batch, outputs):
                if isinstance(output, Exception):
                    print(f"[ERROR] id={question.id}: {output}")
                    continue
                results.append(output.submission)
            write_results(results, args.output)
    finally:
        pipeline.close()

    elapsed = time.perf_counter() - started
    print(f"Hoàn tất {len(results)}/{len(questions)} kết quả trong {elapsed:.1f}s.")


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size phải >= 1.")
    if args.input:
        run_file(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
