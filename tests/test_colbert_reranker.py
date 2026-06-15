import numpy as np

from src.retrieval.colbert_reranker import colbert_score


def test_colbert_score_prefers_matching_document():
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    matching = np.array([[1.0, 0.0], [0.0, 1.0]])
    unrelated = np.array([[-1.0, 0.0], [0.0, -1.0]])

    assert colbert_score(query, matching) > colbert_score(query, unrelated)
