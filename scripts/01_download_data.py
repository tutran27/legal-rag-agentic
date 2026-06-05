from src.data.download_hf import save_dataset_to_parquet, download_parquet_direct


def main():
    # Pháp điển
    for config in [
        "articles",
        "subjects",
        "tree_nodes",
        "ontology_topics",
        "ontology_subjects",
    ]:
        save_dataset_to_parquet(
            hf_id="tmquan/phapdien-moj-gov-vn",
            config=config,
            split="train",
            output_dir="data/raw/phapdien",
        )

    # VBPL large: metadata và relationships có thể dùng load_dataset
    for config in ["metadata", "relationships"]:
        save_dataset_to_parquet(
            hf_id="th1nhng0/vietnamese-legal-documents",
            config=config,
            split="data",
            output_dir="data/raw/vbpl_large",
        )

    # VBPL large: content quá lớn, tải trực tiếp parquet, không qua load_dataset
    download_parquet_direct(
        hf_id="th1nhng0/vietnamese-legal-documents",
        filename="data/content.parquet",
        output_dir="data/raw/vbpl_large",
        output_name="content.parquet",
    )

    # VBPL markdown
    save_dataset_to_parquet(
        hf_id="tmquan/vbpl-vn",
        config=None,
        split="train",
        output_dir="data/raw/vbpl_markdown",
    )

    # Legal instruction
    save_dataset_to_parquet(
        hf_id="duyet/vietnamese-legal-instruct",
        config=None,
        split="train",
        output_dir="data/raw/legal_instruct",
    )


if __name__ == "__main__":
    main()