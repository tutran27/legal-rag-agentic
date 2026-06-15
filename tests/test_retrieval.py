import pandas as pd
import numpy as np
from qdrant_client import models
from types import SimpleNamespace

from src.retrieval.context_expander import expand_context
from src.retrieval.cross_encoder_rerank import cross_encoder_rerank
from src.retrieval.fusion import rrf_fusion
from src.retrieval import hybrid_retriever
from src.schema.agent_schemas import Evidence


class FakeCrossEncoder:
    def predict(self, pairs, batch_size):
        assert batch_size == 16
        return [-5.0, -1.0]


def test_weighted_rrf_prioritizes_original_query():
    original = [
        Evidence(unit_id="original", chunk_id="c1", text="Original")
    ]
    keyword = [
        Evidence(unit_id="keyword", chunk_id="c2", text="Keyword")
    ]

    results = rrf_fusion(
        [original, keyword],
        weights=[1.0, 0.5],
    )

    assert results[0].unit_id == "original"


def test_cross_encoder_handles_negative_scores():
    candidates = [
        Evidence(
            unit_id="u1",
            text="Lower",
            final_score=0.5,
            colbert_normalized_score=0.2,
        ),
        Evidence(
            unit_id="u2",
            text="Higher",
            final_score=0.4,
            colbert_normalized_score=0.8,
        ),
    ]

    results = cross_encoder_rerank(
        "Question",
        candidates,
        model=FakeCrossEncoder(),
    )

    assert results[0].unit_id == "u2"
    assert results[0].cross_encoder_normalized_score == 1.0
    assert results[1].cross_encoder_normalized_score == 0.0


def test_hybrid_search_batches_queries(monkeypatch):
    encoded = []

    def fake_embed_dense(queries, model, is_query):
        encoded.append(queries)
        return np.asarray([[0.1, 0.2], [0.3, 0.4]])

    class FakeClient:
        def __init__(self, **kwargs):
            pass

        def query_batch_points(self, collection_name, requests, timeout):
            assert len(requests) == 2
            return [
                SimpleNamespace(
                    points=[
                        SimpleNamespace(
                            payload={
                                "unit_id": f"u{index}",
                                "chunk_id": f"c{index}",
                                "text": "Text",
                            },
                            score=1.0,
                        )
                    ]
                )
                for index in range(2)
            ]

        def close(self):
            pass

    monkeypatch.setattr(hybrid_retriever, "embed_dense", fake_embed_dense)
    monkeypatch.setattr(hybrid_retriever, "QdrantClient", FakeClient)

    results = hybrid_retriever.hybrid_search_batch(
        ["query one", "query two"],
        dense_st=object(),
    )

    assert encoded == [["query one", "query two"]]
    assert [items[0].unit_id for items in results] == ["u0", "u1"]


def test_hybrid_search_reuses_injected_client_and_dense_vectors(monkeypatch):
    class FakeClient:
        def query_batch_points(self, collection_name, requests, timeout):
            return [SimpleNamespace(points=[])]

    monkeypatch.setattr(
        hybrid_retriever,
        "embed_dense",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Không được encode lại dense vectors.")
        ),
    )

    results = hybrid_retriever.hybrid_search_batch(
        ["query"],
        dense_st=object(),
        client=FakeClient(),
        dense_vectors=np.asarray([[0.1, 0.2]]),
    )

    assert results == [[]]


def test_hybrid_search_accepts_filter_per_query():
    first_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="doc_code",
                match=models.MatchValue(value="A"),
            )
        ]
    )
    second_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="doc_code",
                match=models.MatchValue(value="B"),
            )
        ]
    )

    class FakeClient:
        def query_batch_points(self, collection_name, requests, timeout):
            assert requests[0].prefetch[0].filter == first_filter
            assert requests[1].prefetch[0].filter == second_filter
            return [
                SimpleNamespace(points=[]),
                SimpleNamespace(points=[]),
            ]

    results = hybrid_retriever.hybrid_search_batch(
        ["one", "two"],
        dense_st=object(),
        client=FakeClient(),
        dense_vectors=np.asarray([[0.1, 0.2], [0.3, 0.4]]),
        filters=[first_filter, second_filter],
    )

    assert results == [[], []]


def test_context_expansion_only_keeps_adjacent_parts(tmp_path):
    corpus_path = tmp_path / "corpus.parquet"
    pd.DataFrame(
        [
            {
                "unit_id": "u1",
                "chunk_id": "c0",
                "part_index": 0,
                "text": "Phần trước về điều kiện hỗ trợ",
                "content_text": "Điều kiện hỗ trợ",
            },
            {
                "unit_id": "u1",
                "chunk_id": "c1",
                "part_index": 1,
                "text": "Phần seed",
                "content_text": "Phần seed",
            },
            {
                "unit_id": "u1",
                "chunk_id": "c3",
                "part_index": 3,
                "text": "Phần ở xa",
                "content_text": "Phần ở xa",
            },
        ]
    ).to_parquet(corpus_path, index=False)
    seed = Evidence(
        unit_id="u1",
        chunk_id="c1",
        text="Phần seed",
        final_score=1.0,
        metadata={"part_index": 1},
    )

    results = expand_context(
        [seed],
        query="điều kiện hỗ trợ",
        corpus_path=str(corpus_path),
    )

    assert [item.chunk_id for item in results] == ["c0"]


def test_context_expansion_uses_qdrant_payload():
    class FakeClient:
        def scroll(self, **kwargs):
            return (
                [
                    SimpleNamespace(
                        payload={
                            "unit_id": "u1",
                            "chunk_id": "c0",
                            "part_index": 0,
                            "text": "Điều kiện hỗ trợ",
                            "content_text": "Điều kiện hỗ trợ",
                        }
                    ),
                    SimpleNamespace(
                        payload={
                            "unit_id": "u1",
                            "chunk_id": "c1",
                            "part_index": 1,
                            "text": "Seed",
                            "content_text": "Seed",
                        }
                    ),
                ],
                None,
            )

    seed = Evidence(
        unit_id="u1",
        chunk_id="c1",
        text="Seed",
        final_score=1.0,
        metadata={"part_index": 1, "part_count": 2},
    )

    results = expand_context(
        [seed],
        query="điều kiện hỗ trợ",
        client=FakeClient(),
    )

    assert [item.chunk_id for item in results] == ["c0"]
