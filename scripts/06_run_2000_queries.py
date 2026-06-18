import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter, ValidationError


os.environ.setdefault("LLM_BACKEND", "groq")
os.environ.setdefault("LOCAL_LLM_MODEL", "Qwen/Qwen3-4B-Instruct-2507")
os.environ.setdefault("LOCAL_LLM_MAX_MODEL_LEN", "4096")
os.environ.setdefault("LOCAL_LLM_LOAD_IN_4BIT", "true")
os.environ.setdefault("COLBERT_BATCH_SIZE", "4")
os.environ.setdefault("CROSS_ENCODER_BATCH_SIZE", "4")
os.environ.setdefault("RERANK_MAX_CHARS", "600")
os.environ.setdefault("RETRIEVAL_TOP_K", "40")
os.environ.setdefault("INITIAL_FUSION_TOP_K", "40")
os.environ.setdefault("COLBERT_TOP_K", "20")
os.environ.setdefault("CROSS_ENCODER_TOP_K", "10")
os.environ.setdefault("FINAL_TOP_K", "8")
os.environ.setdefault("PRELOAD_GRAPH", "true")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

from src.schema.agent_schemas import SubmissionItem, TestQuestion
from src.submission.build_results import write_results


DEFAULT_INPUT = "R2AIStage1DATA.json"
DEFAULT_OUTPUT = "results.json"
DEFAULT_ERRORS = "inference_errors.json"
FALLBACK_ANSWER = "Chưa có đủ căn cứ pháp lý liên quan để trả lời câu hỏi."
INPUT_LIST_KEYS = ("questions", "data", "items")


class PipelineLike(Protocol):
    def run(self, question: str, question_id: int): ...

    def close(self) -> None: ...


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
        data = next(
            (data.get(key) for key in INPUT_LIST_KEYS if data.get(key)),
            None,
        )
    if not isinstance(data, list):
        raise ValueError("Input phải là list hoặc dict có questions/data/items.")
    questions = TypeAdapter(list[TestQuestion]).validate_python(data)
    return questions[:limit] if limit else questions


def load_results(path: str | Path) -> dict[int, SubmissionItem]:
    result_path = Path(path)
    if not result_path.exists():
        return {}
    data = json.loads(result_path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, list):
        raise ValueError("Output hiện có phải là list JSON.")
    items = TypeAdapter(list[SubmissionItem]).validate_python(data)
    return {item.id: item for item in items}


def load_errors(path: str | Path) -> list[dict]:
    error_path = Path(path)
    if not error_path.exists():
        return []
    data = json.loads(error_path.read_text(encoding="utf-8-sig"))
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
        answer=FALLBACK_ANSWER,
        relevant_docs=[],
        relevant_articles=[],
    )


def build_pipeline(args: argparse.Namespace) -> PipelineLike:
    from src.generation.endpoint import create_llm_client
    from src.pipeline import InferencePipeline

    llm = create_llm_client(args.llm, args.local_model)
    return InferencePipeline(llm=llm, verbose=False)


def pending_questions(
    questions: list[TestQuestion],
    results: dict[int, SubmissionItem],
) -> list[TestQuestion]:
    return [item for item in questions if item.id not in results]


def save_progress(
    questions: list[TestQuestion],
    results: dict[int, SubmissionItem],
    errors: list[dict],
    output_path: str | Path,
    error_path: str | Path,
) -> None:
    write_ordered_results(questions, results, output_path)
    write_errors(errors, error_path)


def record_error(
    question: TestQuestion,
    error: Exception,
    results: dict[int, SubmissionItem],
    errors: list[dict],
) -> None:
    error_text = str(error)
    results[question.id] = fallback_submission(question, error)
    errors.append(
        {
            "id": question.id,
            "question": question.question,
            "error": error_text,
        }
    )
    print(f"[ERROR DETAIL] id={question.id}: {error_text}")


def log_query_done(
    status: str,
    index: int,
    total_pending: int,
    question_id: int,
    elapsed: float,
    done: int,
    total_questions: int,
) -> None:
    print(
        f"[{status}] {index}/{total_pending} | "
        f"id={question_id} | {elapsed:.1f}s | done={done}/{total_questions}"
    )


def run_one_question(
    pipeline: PipelineLike,
    question: TestQuestion,
    results: dict[int, SubmissionItem],
    errors: list[dict],
) -> str:
    try:
        output = pipeline.run(question.question, question_id=question.id)
        results[question.id] = output.submission
        return "OK"
    except Exception as error:
        record_error(question, error, results, errors)
        return "ERROR"


def run_batch(args: argparse.Namespace) -> None:
    questions = load_questions(args.input, args.limit)
    results = load_results(args.output) if args.resume else {}
    errors = load_errors(args.errors) if args.resume else []
    pending = pending_questions(questions, results)

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
            status = run_one_question(pipeline, question, results, errors)
            save_progress(questions, results, errors, args.output, args.errors)
            elapsed = time.perf_counter() - query_started
            log_query_done(
                status=status,
                index=index,
                total_pending=len(pending),
                question_id=question.id,
                elapsed=elapsed,
                done=len(results),
                total_questions=len(questions),
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
