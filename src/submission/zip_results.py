from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def zip_results(
    input_path: str | Path = "results.json",
    output_path: str | Path = "results.zip",
) -> Path:
    source = Path(input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Không tìm thấy {source}.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(source, arcname="results.json")
    return output
