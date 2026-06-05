from pathlib import Path


def test_phapdien_articles_downloaded():
    path = Path("data/raw/phapdien/articles_train.parquet")
    assert path.exists()
    assert path.stat().st_size > 0


def test_vbpl_metadata_downloaded():
    path = Path("data/raw/vbpl_large/metadata_data.parquet")
    assert path.exists()
    assert path.stat().st_size > 0


def test_vbpl_content_downloaded():
    path = Path("data/raw/vbpl_large/content_data.parquet")
    assert path.exists()
    assert path.stat().st_size > 0

if __name__ == "__main__":
    test_phapdien_articles_downloaded()
    test_vbpl_metadata_downloaded()
    test_vbpl_content_downloaded()
    print("All tests passed!")