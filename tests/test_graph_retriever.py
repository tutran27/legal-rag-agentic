import pickle

import networkx as nx
import pandas as pd

from src.retrieval.graph_retriever import graph_search
from src.schema.agent_schemas import Evidence


def test_graph_search_returns_related_document(tmp_path):
    graph = nx.MultiDiGraph()
    graph.add_edge("1", "2", relation_type="GUIDES")

    graph_path = tmp_path / "graph.pkl"
    with graph_path.open("wb") as file:
        pickle.dump(graph, file)

    corpus_path = tmp_path / "corpus.parquet"
    pd.DataFrame(
        [
            {
                "unit_id": "unit-2",
                "chunk_id": "chunk-2",
                "chunk_type": "article_part",
                "text": "Nội dung liên quan",
                "doc_id": "2",
                "doc_code": "02/2025/NĐ-CP",
                "doc_title_submission": "Nghị định 02/2025/NĐ-CP",
                "article": "Điều 1",
                "article_title": "Phạm vi",
                "status": "Còn hiệu lực",
                "is_current": True,
            }
        ]
    ).to_parquet(corpus_path, index=False)

    seed = Evidence(
        unit_id="unit-1",
        text="Seed",
        metadata={"doc_id": "1"},
    )
    results = graph_search(
        "nội dung liên quan",
        [seed],
        graph_path=str(graph_path),
        corpus_path=str(corpus_path),
    )

    assert results[0].unit_id == "unit-2"
    assert results[0].metadata["relation_type"] == "GUIDES"
