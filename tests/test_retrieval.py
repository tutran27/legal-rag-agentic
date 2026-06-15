import pandas as pd

from src.retrieval.context_expander import expand_context
from src.retrieval.cross_encoder_rerank import cross_encoder_rerank
from src.retrieval.fusion import rrf_fusion
from src.schema.agent_schemas import Evidence


class FakeCrossEncoder:
    def predict(self, pairs, batch_size):
        assert batch_size == 8
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
