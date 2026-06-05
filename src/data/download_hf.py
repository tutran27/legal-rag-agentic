from pathlib import Path
import shutil
from huggingface_hub import hf_hub_download
from datasets import load_dataset


def save_dataset_to_parquet(
    hf_id: str,
    config: str | None,
    split: str,
    output_dir: str,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    name = config if config is not None else "default"
    output_path = output_dir / f"{name}.parquet"

    print(f"Loading dataset: {hf_id}, config={config}, split={split}")

    if config is None:
        ds = load_dataset(hf_id, split=split)
    else:
        ds = load_dataset(hf_id, config, split=split)

    ds.to_parquet(str(output_path))
    print(f"Saved to {output_path}")


def download_parquet_direct(
    hf_id: str,
    filename: str,
    output_dir: str,
    output_name: str | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_path = hf_hub_download(
        repo_id=hf_id,
        repo_type="dataset",
        filename=filename,
    )

    output_path = output_dir / (output_name or Path(filename).name)
    shutil.copy2(local_path, output_path)

    print(f"Downloaded direct parquet: {hf_id}/{filename}")
    print(f"Saved to {output_path}")

    return output_path