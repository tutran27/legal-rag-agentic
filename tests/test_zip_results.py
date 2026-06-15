from zipfile import ZipFile

from src.submission.zip_results import zip_results


def test_zip_results_is_flat(tmp_path):
    source = tmp_path / "results.json"
    source.write_text("[]", encoding="utf-8")

    output = zip_results(source, tmp_path / "results.zip")

    with ZipFile(output) as archive:
        assert archive.namelist() == ["results.json"]
