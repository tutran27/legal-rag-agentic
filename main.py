import argparse
import json
import sys
import time
from pathlib import Path

from pydantic import TypeAdapter

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.generation.endpoint import create_llm_client
from src.pipeline import InferencePipeline
from src.schema.agent_schemas import SubmissionItem, TestQuestion
from src.submission.build_results import write_results


DEFAULT_QUERY = (
    "Doanh nghiệp nhỏ và vừa phải đáp ứng điều kiện nào để được hỗ trợ "
    "theo Luật Hỗ trợ doanh nghiệp nhỏ và vừa?"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Chạy legal RAG pipeline.")
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--question-id", type=int, default=1)
    parser.add_argument("--input")
    parser.add_argument("--output", default="results.json")
    parser.add_argument("--errors", default="inference_errors.json")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--llm",
        choices=["endpoint", "local"],
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
    return TypeAdapter(list[TestQuestion]).validate_python(data)


def load_existing_results(path: str | Path) -> list[SubmissionItem]:
    output_path = Path(path)
    if not output_path.exists():
        return []
    data = json.loads(output_path.read_text(encoding="utf-8-sig"))
    return TypeAdapter(list[SubmissionItem]).validate_python(data)


def run_batch(args: argparse.Namespace) -> None:
    questions = load_questions(args.input)
    if args.limit is not None:
        questions = questions[:args.limit]

    completed = {
        item.id: item for item in load_existing_results(args.output)
    }
    errors = []
    llm = create_llm_client(args.llm, args.local_model)
    pipeline = InferencePipeline(llm=llm, verbose=not args.quiet)
    batch_started = time.perf_counter()
    processed = 0

    try:
        pending = [
            item
            for item in questions
            if (
                item.id not in completed
                or completed[item.id].question != item.question
            )
        ]
        batch_size = max(args.batch_size, 1)
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            batch_started_at = time.perf_counter()
            for item in batch:
                print(f"Chạy ID {item.id}: {item.question}")

            batch_outputs = pipeline.run_many(
                [(item.id, item.question) for item in batch]
            )

            for item, output in zip(batch, batch_outputs):
                if isinstance(output, Exception):
                    errors.append(
                        {
                            "id": item.id,
                            "question": item.question,
                            "error": str(output),
                        }
                    )
                    print(f"[ERROR] ID {item.id}: {output}")
                    continue
                completed[item.id] = output.submission
                processed += 1
                print(f"[OK] ID {item.id}")

            write_results(
                [completed[key] for key in sorted(completed)],
                args.output,
            )
            if errors:
                Path(args.errors).write_text(
                    json.dumps(errors, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            elapsed = time.perf_counter() - batch_started_at
            average = (time.perf_counter() - batch_started) / max(
                processed,
                1,
            )
            remaining = len(pending) - min(
                start + batch_size,
                len(pending),
            )
            print(
                f"[BATCH] {len(batch)} câu: {elapsed:.1f}s | "
                f"ETA khoảng {remaining * average / 3600:.1f} giờ"
            )
    finally:
        pipeline.close()

    print(
        f"Hoàn thành {len(completed)}/{len(questions)} câu. "
        f"Kết quả: {Path(args.output).resolve()}"
    )
    if errors:
        print(f"Có {len(errors)} lỗi: {Path(args.errors).resolve()}")


def run_single(args: argparse.Namespace) -> None:
    llm = create_llm_client(args.llm, args.local_model)
    pipeline = InferencePipeline(llm=llm, verbose=not args.quiet)
    try:
        result = pipeline.run(args.query, question_id=args.question_id)
    finally:
        pipeline.close()
    output_path = write_results([result.submission], args.output)

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


def main() -> None:
    args = parse_args()
    if args.input:
        run_batch(args)
    else:
        run_single(args)


if __name__ == "__main__":
    main()
