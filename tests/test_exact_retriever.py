from types import SimpleNamespace

from src.retrieval.exact_retriever import exact_search


def test_exact_search_uses_qdrant_when_client_is_provided():
    class FakeClient:
        def scroll(self, **kwargs):
            return (
                [
                    SimpleNamespace(
                        payload={
                            "unit_id": "u1",
                            "chunk_id": "c1",
                            "text": "Nội dung Điều 12",
                            "doc_code": "04/2017/QH14",
                            "article": "Điều 12",
                        }
                    )
                ],
                None,
            )

    results = exact_search(
        "Điều 12 Luật 04/2017/QH14",
        doc_codes=["04/2017/QH14"],
        client=FakeClient(),
    )

    assert results[0].unit_id == "u1"
    assert results[0].source == "exact"
