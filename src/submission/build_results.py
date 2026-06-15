import json
from pathlib import Path

from src.schema.agent_schemas import SubmissionItem


def write_results(
    items: list[SubmissionItem],
    output_path: str | Path = "results.json",
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            [item.model_dump() for item in items],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
