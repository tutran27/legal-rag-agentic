import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.submission.validate_results import load_and_validate_results
from src.submission.zip_results import zip_results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kiểm tra results.json và tạo results.zip phẳng."
    )
    parser.add_argument("--input", default="results.json")
    parser.add_argument("--output", default="results.zip")
    args = parser.parse_args()

    items = load_and_validate_results(args.input)
    output = zip_results(args.input, args.output)
    print(f"Validated {len(items)} result(s).")
    print(f"Created: {output.resolve()}")


if __name__ == "__main__":
    main()
